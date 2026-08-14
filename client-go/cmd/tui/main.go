// medicalai-tui is a terminal UI client for the MedicalAI - Light Weight
// REST API. Run the server first (from the repo root):
//
//	python quantization.py --mode serve-api
//
// then, from client-go/:
//
//	go run ./cmd/tui
//
// This process does no ML work itself; every action is an HTTP call to
// the Python server, which is what keeps this a small, fast-starting
// static binary instead of needing torch/transformers on the client
// machine too.
package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/bubbles/textinput"
	"github.com/charmbracelet/lipgloss"

	"medicalai-client/internal/api"
)

type screen int

const (
	screenMenu screen = iota
	screenImagePath
	screenSymptoms
	screenMeds
	screenResult
	screenLoading
)

type action int

const (
	actionNone action = iota
	actionHealth
	actionVision
	actionSymptomCheck
	actionMeds
)

var (
	titleStyle   = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("39"))
	dimStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	errorStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("196"))
	successStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("42"))
	warnStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("214"))
	cursorStyle  = lipgloss.NewStyle().Foreground(lipgloss.Color("39"))
)

var menuItems = []string{
	"System Health & Status",
	"Vision — Generate report from X-ray",
	"Symptom Check — Diagnose from X-ray + symptoms",
	"Medication Interaction Check",
	"Quit",
}

type model struct {
	client      *api.Client
	screen      screen
	menuCursor  int
	pendingAct  action
	input       textinput.Model
	imagePath   string
	resultText  string
	resultIsErr bool
	quitting    bool
}

func initialModel(baseURL string) model {
	ti := textinput.New()
	ti.Placeholder = "..."
	ti.CharLimit = 512
	ti.Width = 60
	return model{
		client: api.New(baseURL),
		screen: screenMenu,
		input:  ti,
	}
}

func (m model) Init() tea.Cmd {
	return nil
}

type resultMsg struct {
	text  string
	isErr bool
}

func doHealth(c *api.Client) tea.Cmd {
	return func() tea.Msg {
		h, err := c.Health()
		if err != nil {
			return resultMsg{text: err.Error(), isErr: true}
		}
		var b strings.Builder
		fmt.Fprintf(&b, "Status:       %s\n", h.Status)
		fmt.Fprintf(&b, "Device:       %s\n", h.Device)
		fmt.Fprintf(&b, "Model source: %s\n", h.ModelSource)
		fmt.Fprintf(&b, "CPU threads:  %d\n", h.CPUThreads)
		fmt.Fprintf(&b, "ONNX providers: %s\n", strings.Join(h.ONNXProviders, ", "))
		if rss, ok := h.Memory["rss_mb"]; ok {
			fmt.Fprintf(&b, "Process RAM:  %.0f MB\n", toFloat(rss))
		}
		return resultMsg{text: b.String()}
	}
}

func toFloat(v any) float64 {
	f, _ := v.(float64)
	return f
}

func doVision(c *api.Client, imagePath string) tea.Cmd {
	return func() tea.Msg {
		r, err := c.Vision(imagePath)
		if err != nil {
			return resultMsg{text: err.Error(), isErr: true}
		}
		return resultMsg{text: fmt.Sprintf("Caption: %s\n\n(%.0f ms)", r.Caption, r.InferenceTimeMs)}
	}
}

func doSymptomCheck(c *api.Client, imagePath, symptoms string) tea.Cmd {
	return func() tea.Msg {
		r, err := c.SymptomCheck(imagePath, symptoms, nil)
		if err != nil {
			return resultMsg{text: err.Error(), isErr: true}
		}
		var b strings.Builder
		fmt.Fprintf(&b, "Diagnosis:  %s\n", r.Diagnosis)
		fmt.Fprintf(&b, "Confidence: %.1f%%\n\n", r.Confidence*100)
		fmt.Fprintf(&b, "%s\n\n", r.Description)
		fmt.Fprintf(&b, "Urgency: %s\nFollow-up: %s\n", r.Urgency, r.FollowUp)
		fmt.Fprintf(&b, "\n(%.0f ms)", r.InferenceTimeMs)
		return resultMsg{text: b.String()}
	}
}

func doMedsCheck(c *api.Client, medsText string) tea.Cmd {
	return func() tea.Msg {
		var meds []string
		for _, part := range strings.FieldsFunc(medsText, func(r rune) bool { return r == ',' || r == '\n' }) {
			part = strings.TrimSpace(part)
			if part != "" {
				meds = append(meds, part)
			}
		}
		if len(meds) < 2 {
			return resultMsg{text: "Enter at least 2 medications, separated by commas.", isErr: true}
		}
		r, err := c.CheckInteractions(meds)
		if err != nil {
			return resultMsg{text: err.Error(), isErr: true}
		}
		var b strings.Builder
		if len(r.Interactions) == 0 {
			b.WriteString("No known interactions found among these medications.\n\n")
		}
		for _, it := range r.Interactions {
			fmt.Fprintf(&b, "%s + %s: %s", it.DrugA, it.DrugB, it.Level)
			if it.Note != "" {
				fmt.Fprintf(&b, " — %s", it.Note)
			}
			fmt.Fprintf(&b, " (%s)\n", it.Source)
		}
		if len(r.Unrecognized) > 0 {
			fmt.Fprintf(&b, "\nNot found in the interaction database: %s\n", strings.Join(r.Unrecognized, ", "))
		}
		if !r.FullDatabaseUsed {
			b.WriteString("\n(Using the small built-in reference set only — run 'python drug_interactions.py --download' server-side for full coverage.)\n")
		}
		fmt.Fprintf(&b, "\n%s", r.Disclaimer)
		return resultMsg{text: b.String()}
	}
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.KeyMsg:
		switch m.screen {
		case screenMenu:
			return m.updateMenu(msg)
		case screenImagePath, screenSymptoms, screenMeds:
			return m.updateInput(msg)
		case screenResult:
			if msg.String() == "enter" || msg.String() == "esc" || msg.String() == "q" {
				m.screen = screenMenu
				return m, nil
			}
		case screenLoading:
			// ignore key presses while a request is in flight
		}
	case resultMsg:
		m.resultText = msg.text
		m.resultIsErr = msg.isErr
		m.screen = screenResult
		return m, nil
	}
	return m, nil
}

func (m model) updateMenu(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c", "q":
		m.quitting = true
		return m, tea.Quit
	case "up", "k":
		if m.menuCursor > 0 {
			m.menuCursor--
		}
	case "down", "j":
		if m.menuCursor < len(menuItems)-1 {
			m.menuCursor++
		}
	case "enter":
		switch m.menuCursor {
		case 0:
			m.screen = screenLoading
			return m, doHealth(m.client)
		case 1:
			m.pendingAct = actionVision
			m.screen = screenImagePath
			m.input.SetValue("")
			m.input.Placeholder = "/path/to/xray.jpg"
			m.input.Focus()
			return m, textinput.Blink
		case 2:
			m.pendingAct = actionSymptomCheck
			m.screen = screenImagePath
			m.input.SetValue("")
			m.input.Placeholder = "/path/to/xray.jpg"
			m.input.Focus()
			return m, textinput.Blink
		case 3:
			m.pendingAct = actionMeds
			m.screen = screenMeds
			m.input.SetValue("")
			m.input.Placeholder = "Warfarin, Ibuprofen, Metoprolol"
			m.input.Focus()
			return m, textinput.Blink
		case 4:
			m.quitting = true
			return m, tea.Quit
		}
	}
	return m, nil
}

func (m model) updateInput(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c":
		m.quitting = true
		return m, tea.Quit
	case "esc":
		m.screen = screenMenu
		return m, nil
	case "enter":
		value := strings.TrimSpace(m.input.Value())
		if value == "" {
			return m, nil
		}
		switch m.screen {
		case screenImagePath:
			m.imagePath = value
			if m.pendingAct == actionVision {
				m.screen = screenLoading
				return m, doVision(m.client, m.imagePath)
			}
			// symptom check needs one more field
			m.screen = screenSymptoms
			m.input.SetValue("")
			m.input.Placeholder = "shortness of breath, cough, fever..."
			return m, nil
		case screenSymptoms:
			m.screen = screenLoading
			return m, doSymptomCheck(m.client, m.imagePath, value)
		case screenMeds:
			m.screen = screenLoading
			return m, doMedsCheck(m.client, value)
		}
	}
	var cmd tea.Cmd
	m.input, cmd = m.input.Update(msg)
	return m, cmd
}

func (m model) View() string {
	if m.quitting {
		return "Bye.\n"
	}

	var b strings.Builder
	b.WriteString(titleStyle.Render("MedicalAI - Light Weight (TUI)") + "\n")
	b.WriteString(dimStyle.Render("Server: "+m.client.BaseURL) + "\n\n")

	switch m.screen {
	case screenMenu:
		for i, item := range menuItems {
			cursor := "  "
			if i == m.menuCursor {
				cursor = cursorStyle.Render("> ")
			}
			b.WriteString(cursor + item + "\n")
		}
		b.WriteString("\n" + dimStyle.Render("↑/↓ move · enter select · q quit"))
	case screenImagePath:
		b.WriteString("Image path:\n" + m.input.View() + "\n\n" + dimStyle.Render("enter confirm · esc back"))
	case screenSymptoms:
		b.WriteString("Patient symptoms / clinical indication:\n" + m.input.View() + "\n\n" + dimStyle.Render("enter confirm · esc back"))
	case screenMeds:
		b.WriteString("Current medications (comma-separated):\n" + m.input.View() + "\n\n" + dimStyle.Render("enter confirm · esc back"))
	case screenLoading:
		b.WriteString("Working...\n")
	case screenResult:
		style := successStyle
		if m.resultIsErr {
			style = errorStyle
		}
		b.WriteString(style.Render(m.resultText) + "\n\n")
		b.WriteString(dimStyle.Render("enter/esc back to menu"))
	}

	b.WriteString("\n\n" + warnStyle.Render("General education, not a medical diagnosis. Always confirm with a clinician."))
	return b.String()
}

func main() {
	server := flag.String("server", "http://127.0.0.1:8000", "MedicalAI API server URL")
	flag.Parse()

	p := tea.NewProgram(initialModel(*server))
	if _, err := p.Run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}
