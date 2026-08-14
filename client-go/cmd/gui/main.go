// medicalai-gui is a native desktop client for the MedicalAI - Light
// Weight REST API, built with Fyne. Like the TUI, it does no ML work
// itself — every action is an HTTP call to the Python server started
// with `python quantization.py --mode serve-api`. Run that first.
package main

import (
	"flag"
	"fmt"
	"strings"

	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/dialog"
	"fyne.io/fyne/v2/widget"

	"medicalai-client/internal/api"
)

const disclaimer = "General education, not a medical diagnosis. Always confirm with a clinician."

func main() {
	server := flag.String("server", "http://127.0.0.1:8000", "MedicalAI API server URL")
	flag.Parse()

	client := api.New(*server)

	a := app.New()
	w := a.NewWindow("MedicalAI - Light Weight")
	w.Resize(fyne.NewSize(720, 560))

	tabs := container.NewAppTabs(
		container.NewTabItem("Vision", visionTab(w, client)),
		container.NewTabItem("Symptom Check", symptomTab(w, client)),
		container.NewTabItem("Medication Check", medsTab(client)),
		container.NewTabItem("Status", statusTab(client)),
	)

	footer := widget.NewLabel(disclaimer)
	footer.Wrapping = fyne.TextWrapWord

	w.SetContent(container.NewBorder(nil, footer, nil, nil, tabs))
	w.ShowAndRun()
}

// imagePicker returns a label showing the selected path, a button that
// opens a file dialog, and a func to read the currently selected path.
func imagePicker(w fyne.Window) (fyne.CanvasObject, func() string) {
	pathLabel := widget.NewLabel("No image selected.")
	pathLabel.Wrapping = fyne.TextWrapBreak
	selected := ""

	btn := widget.NewButton("Choose X-ray Image...", func() {
		dialog.ShowFileOpen(func(reader fyne.URIReadCloser, err error) {
			if err != nil || reader == nil {
				return
			}
			defer reader.Close()
			selected = reader.URI().Path()
			pathLabel.SetText(selected)
		}, w)
	})

	row := container.NewVBox(btn, pathLabel)
	return row, func() string { return selected }
}

func visionTab(w fyne.Window, client *api.Client) fyne.CanvasObject {
	picker, getPath := imagePicker(w)
	output := widget.NewMultiLineEntry()
	output.Wrapping = fyne.TextWrapWord
	output.SetPlaceHolder("Generated report will appear here.")

	runBtn := widget.NewButton("Generate Report", func() {
		path := getPath()
		if path == "" {
			output.SetText("Choose an image first.")
			return
		}
		output.SetText("Working...")
		go func() {
			r, err := client.Vision(path)
			if err != nil {
				output.SetText("Error: " + err.Error())
				return
			}
			output.SetText(fmt.Sprintf("%s\n\n(%.0f ms)", r.Caption, r.InferenceTimeMs))
		}()
	})

	return container.NewVBox(
		widget.NewLabel("Upload a chest X-ray to generate a radiology caption."),
		picker,
		runBtn,
		widget.NewSeparator(),
		output,
	)
}

func symptomTab(w fyne.Window, client *api.Client) fyne.CanvasObject {
	picker, getPath := imagePicker(w)
	symptoms := widget.NewMultiLineEntry()
	symptoms.SetPlaceHolder("e.g., shortness of breath, cough, fever...")

	output := widget.NewMultiLineEntry()
	output.Wrapping = fyne.TextWrapWord
	output.SetPlaceHolder("Diagnosis result will appear here.")

	runBtn := widget.NewButton("Diagnose", func() {
		path := getPath()
		if path == "" {
			output.SetText("Choose an image first.")
			return
		}
		text := symptoms.Text
		if strings.TrimSpace(text) == "" {
			text = "No symptoms provided"
		}
		output.SetText("Working...")
		go func() {
			r, err := client.SymptomCheck(path, text, nil)
			if err != nil {
				output.SetText("Error: " + err.Error())
				return
			}
			output.SetText(fmt.Sprintf(
				"Diagnosis: %s\nConfidence: %.1f%%\n\n%s\n\nUrgency: %s\nFollow-up: %s\n\n(%.0f ms)",
				r.Diagnosis, r.Confidence*100, r.Description, r.Urgency, r.FollowUp, r.InferenceTimeMs,
			))
		}()
	})

	return container.NewVBox(
		widget.NewLabel("Upload an X-ray and enter symptoms for a fused image+text diagnosis."),
		picker,
		widget.NewLabel("Symptoms:"),
		symptoms,
		runBtn,
		widget.NewSeparator(),
		output,
	)
}

func medsTab(client *api.Client) fyne.CanvasObject {
	meds := widget.NewMultiLineEntry()
	meds.SetPlaceHolder("Warfarin\nIbuprofen\nMetoprolol\n(one per line, or comma-separated)")

	output := widget.NewMultiLineEntry()
	output.Wrapping = fyne.TextWrapWord
	output.SetPlaceHolder("Interaction results will appear here.")

	runBtn := widget.NewButton("Check Interactions", func() {
		var list []string
		for _, part := range strings.FieldsFunc(meds.Text, func(r rune) bool { return r == ',' || r == '\n' }) {
			part = strings.TrimSpace(part)
			if part != "" {
				list = append(list, part)
			}
		}
		if len(list) < 2 {
			output.SetText("Enter at least 2 medications.")
			return
		}
		output.SetText("Working...")
		go func() {
			r, err := client.CheckInteractions(list)
			if err != nil {
				output.SetText("Error: " + err.Error())
				return
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
				b.WriteString("\n(Using the small built-in reference set only.)\n")
			}
			fmt.Fprintf(&b, "\n%s", r.Disclaimer)
			output.SetText(b.String())
		}()
	})

	return container.NewVBox(
		widget.NewLabel("Enter the patient's current medications to screen for known interactions."),
		meds,
		runBtn,
		widget.NewSeparator(),
		output,
	)
}

func statusTab(client *api.Client) fyne.CanvasObject {
	output := widget.NewMultiLineEntry()
	output.Wrapping = fyne.TextWrapWord
	output.Disable()

	refresh := func() {
		go func() {
			h, err := client.Health()
			if err != nil {
				output.SetText("Error: " + err.Error() + "\n\nIs the server running? Start it with:\n  python quantization.py --mode serve-api")
				return
			}
			output.SetText(fmt.Sprintf(
				"Status: %s\nDevice: %s\nModel source: %s\nCPU threads: %d\nONNX providers: %s",
				h.Status, h.Device, h.ModelSource, h.CPUThreads, strings.Join(h.ONNXProviders, ", "),
			))
		}()
	}

	refreshBtn := widget.NewButton("Refresh", refresh)
	refresh()

	return container.NewVBox(
		widget.NewLabel("Server status:"),
		refreshBtn,
		output,
	)
}
