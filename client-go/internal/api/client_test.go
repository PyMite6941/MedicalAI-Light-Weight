package api

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
)

// mockServer stands in for quantization.py's FastAPI server so the client
// can be tested without a real Python/torch environment.
func mockServer(t *testing.T) *httptest.Server {
	t.Helper()
	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(Health{
			Status: "ok", Device: "CPU", ModelSource: "Default (random weights)",
			Memory:        map[string]any{"rss_mb": 123.0},
			Models:        map[string]any{"trained_pytorch": false},
			ONNXProviders: []string{"CPUExecutionProvider"},
			CPUThreads:    4,
		})
	})

	mux.HandleFunc("/api/models", func(w http.ResponseWriter, r *http.Request) {
		json.NewEncoder(w).Encode(map[string]any{"trained_pytorch": false, "default_classifier": true})
	})

	mux.HandleFunc("/api/vision", func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(10 << 20); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if _, _, err := r.FormFile("file"); err != nil {
			http.Error(w, "missing file", 400)
			return
		}
		json.NewEncoder(w).Encode(VisionResult{Caption: "test caption", InferenceTimeMs: 42.5})
	})

	mux.HandleFunc("/api/symptom-check", func(w http.ResponseWriter, r *http.Request) {
		if err := r.ParseMultipartForm(10 << 20); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		symptoms := r.FormValue("symptoms")
		if symptoms == "" {
			http.Error(w, "missing symptoms", 400)
			return
		}
		json.NewEncoder(w).Encode(SymptomResult{
			Diagnosis: "Pneumonia", Confidence: 0.82, InferenceTimeMs: 55.1,
			Description: "test description", Urgency: "urgent", FollowUp: "test follow-up",
		})
	})

	mux.HandleFunc("/api/check-interactions", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Medications []string `json:"medications"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if len(req.Medications) < 2 {
			http.Error(w, "need at least 2 medications", 400)
			return
		}
		json.NewEncoder(w).Encode(InteractionCheckResult{
			Interactions: []Interaction{
				{DrugA: req.Medications[0], DrugB: req.Medications[1], Level: "Major", Source: "test"},
			},
			Unrecognized:     []string{},
			FullDatabaseUsed: true,
			Disclaimer:       "test disclaimer",
		})
	})

	return httptest.NewServer(mux)
}

func TestHealth(t *testing.T) {
	srv := mockServer(t)
	defer srv.Close()
	c := New(srv.URL)

	h, err := c.Health()
	if err != nil {
		t.Fatalf("Health() error: %v", err)
	}
	if h.Device != "CPU" || h.CPUThreads != 4 || len(h.ONNXProviders) != 1 {
		t.Fatalf("unexpected health response: %+v", h)
	}
}

func TestModels(t *testing.T) {
	srv := mockServer(t)
	defer srv.Close()
	c := New(srv.URL)

	m, err := c.Models()
	if err != nil {
		t.Fatalf("Models() error: %v", err)
	}
	if m["default_classifier"] != true {
		t.Fatalf("unexpected models response: %+v", m)
	}
}

func TestVision(t *testing.T) {
	srv := mockServer(t)
	defer srv.Close()
	c := New(srv.URL)

	tmp, err := os.CreateTemp(t.TempDir(), "*.jpg")
	if err != nil {
		t.Fatal(err)
	}
	tmp.WriteString("fake image bytes")
	tmp.Close()

	r, err := c.Vision(tmp.Name())
	if err != nil {
		t.Fatalf("Vision() error: %v", err)
	}
	if r.Caption != "test caption" || r.InferenceTimeMs != 42.5 {
		t.Fatalf("unexpected vision response: %+v", r)
	}
}

func TestSymptomCheck(t *testing.T) {
	srv := mockServer(t)
	defer srv.Close()
	c := New(srv.URL)

	tmp, err := os.CreateTemp(t.TempDir(), "*.jpg")
	if err != nil {
		t.Fatal(err)
	}
	tmp.WriteString("fake image bytes")
	tmp.Close()

	r, err := c.SymptomCheck(tmp.Name(), "cough and fever", nil)
	if err != nil {
		t.Fatalf("SymptomCheck() error: %v", err)
	}
	if r.Diagnosis != "Pneumonia" || r.Urgency != "urgent" {
		t.Fatalf("unexpected symptom-check response: %+v", r)
	}
}

func TestCheckInteractions(t *testing.T) {
	srv := mockServer(t)
	defer srv.Close()
	c := New(srv.URL)

	r, err := c.CheckInteractions([]string{"Warfarin", "Ibuprofen"})
	if err != nil {
		t.Fatalf("CheckInteractions() error: %v", err)
	}
	if len(r.Interactions) != 1 || r.Interactions[0].Level != "Major" {
		t.Fatalf("unexpected interactions response: %+v", r)
	}

	if _, err := c.CheckInteractions([]string{"OnlyOne"}); err == nil {
		t.Fatal("expected error for a single medication, got nil")
	}
}

func TestHealthUnreachable(t *testing.T) {
	c := New("http://127.0.0.1:1") // nothing listens here
	if _, err := c.Health(); err == nil {
		t.Fatal("expected error when server is unreachable, got nil")
	}
}
