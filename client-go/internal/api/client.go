// Package api is a thin HTTP client for the MedicalAI - Light Weight
// REST API (served by `python quantization.py --mode serve-api`, see
// quantization.py in the repo root). It intentionally does no ML work
// itself — model inference stays in Python/PyTorch where the ML
// ecosystem lives; this just talks HTTP to a server that already does
// that. That split is what keeps this client tiny: a static Go binary
// with no Python/torch runtime required wherever the TUI or GUI runs.
package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

// Client talks to a single MedicalAI API server instance.
type Client struct {
	BaseURL string
	HTTP    *http.Client
}

// New returns a Client with a sane default timeout. Inference calls
// (vision captioning especially) can take tens of seconds on CPU, so the
// timeout is generous rather than snappy.
func New(baseURL string) *Client {
	return &Client{
		BaseURL: baseURL,
		HTTP:    &http.Client{Timeout: 120 * time.Second},
	}
}

// Health mirrors quantization.py's HealthResponse model.
type Health struct {
	Status        string         `json:"status"`
	Device        string         `json:"device"`
	ModelSource   string         `json:"model_source"`
	Memory        map[string]any `json:"memory"`
	Models        map[string]any `json:"models"`
	ONNXProviders []string       `json:"onnx_providers"`
	CPUThreads    int            `json:"cpu_threads"`
}

// VisionResult mirrors VisionResponse.
type VisionResult struct {
	Caption         string  `json:"caption"`
	InferenceTimeMs float64 `json:"inference_time_ms"`
}

// SymptomResult mirrors SymptomResponse.
type SymptomResult struct {
	Diagnosis       string  `json:"diagnosis"`
	Confidence      float64 `json:"confidence"`
	InferenceTimeMs float64 `json:"inference_time_ms"`
	Description     string  `json:"description"`
	Urgency         string  `json:"urgency"`
	FollowUp        string  `json:"follow_up"`
}

// Interaction mirrors one entry of InteractionCheckResponse.interactions.
type Interaction struct {
	DrugA  string `json:"drug_a"`
	DrugB  string `json:"drug_b"`
	Level  string `json:"level"`
	Note   string `json:"note"`
	Source string `json:"source"`
}

// InteractionCheckResult mirrors InteractionCheckResponse.
type InteractionCheckResult struct {
	Interactions      []Interaction `json:"interactions"`
	Unrecognized      []string      `json:"unrecognized"`
	FullDatabaseUsed  bool          `json:"full_database_used"`
	Disclaimer        string        `json:"disclaimer"`
}

func (c *Client) get(path string, out any) error {
	resp, err := c.HTTP.Get(c.BaseURL + path)
	if err != nil {
		return fmt.Errorf("request to %s failed: %w (is the server running? see README)", path, err)
	}
	defer resp.Body.Close()
	return decodeOrError(resp, out)
}

func decodeOrError(resp *http.Response, out any) error {
	if resp.StatusCode >= 400 {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("server returned %d: %s", resp.StatusCode, string(body))
	}
	if out == nil {
		return nil
	}
	return json.NewDecoder(resp.Body).Decode(out)
}

// Health calls GET /health.
func (c *Client) Health() (*Health, error) {
	var h Health
	if err := c.get("/health", &h); err != nil {
		return nil, err
	}
	return &h, nil
}

// Models calls GET /api/models.
func (c *Client) Models() (map[string]any, error) {
	var m map[string]any
	if err := c.get("/api/models", &m); err != nil {
		return nil, err
	}
	return m, nil
}

func (c *Client) postImage(path, imagePath string, fields map[string]string, out any) error {
	f, err := os.Open(imagePath)
	if err != nil {
		return fmt.Errorf("could not open image %s: %w", imagePath, err)
	}
	defer f.Close()

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, err := writer.CreateFormFile("file", filepath.Base(imagePath))
	if err != nil {
		return err
	}
	if _, err := io.Copy(part, f); err != nil {
		return err
	}
	for k, v := range fields {
		if err := writer.WriteField(k, v); err != nil {
			return err
		}
	}
	if err := writer.Close(); err != nil {
		return err
	}

	req, err := http.NewRequest(http.MethodPost, c.BaseURL+path, &buf)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", writer.FormDataContentType())

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return fmt.Errorf("request to %s failed: %w", path, err)
	}
	defer resp.Body.Close()
	return decodeOrError(resp, out)
}

// Vision calls POST /api/vision with the given image file.
func (c *Client) Vision(imagePath string) (*VisionResult, error) {
	var v VisionResult
	if err := c.postImage("/api/vision", imagePath, nil, &v); err != nil {
		return nil, err
	}
	return &v, nil
}

// SymptomCheck calls POST /api/symptom-check with the given image and
// symptom text. useONNX == nil leaves the choice to the server's default.
func (c *Client) SymptomCheck(imagePath, symptoms string, useONNX *bool) (*SymptomResult, error) {
	fields := map[string]string{"symptoms": symptoms}
	if useONNX != nil {
		if *useONNX {
			fields["use_onnx"] = "true"
		} else {
			fields["use_onnx"] = "false"
		}
	}
	var s SymptomResult
	if err := c.postImage("/api/symptom-check", imagePath, fields, &s); err != nil {
		return nil, err
	}
	return &s, nil
}

// CheckInteractions calls POST /api/check-interactions with a list of
// medication names.
func (c *Client) CheckInteractions(medications []string) (*InteractionCheckResult, error) {
	body, err := json.Marshal(map[string][]string{"medications": medications})
	if err != nil {
		return nil, err
	}
	resp, err := c.HTTP.Post(c.BaseURL+"/api/check-interactions", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("request to /api/check-interactions failed: %w", err)
	}
	defer resp.Body.Close()
	var result InteractionCheckResult
	if err := decodeOrError(resp, &result); err != nil {
		return nil, err
	}
	return &result, nil
}
