import React, { useState } from "react";

export default function VoiceCheck() {
  const API_URL = "http://localhost:5000/predict";

  const [file, setFile]   = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // handles the <input type="file"> change
  const handleFile = (e) => {
    setFile(e.target.files[0] ?? null);
    setResult(null);
  };

  // POST selected file to Flask
  const submit = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const r = await fetch(API_URL, { method: "POST", body: form });
      const json = await r.json();
      if (r.ok) {
        setResult(json.label);
      } else {
        // server sent error message in json.error
        setResult(`error: ${json.error ?? r.statusText}`);
      }
    } catch (err) {
      console.error(err);
      setResult("network error");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ fontFamily: "sans-serif", maxWidth: 400 }}>
      <h2>Voice Checker</h2>

      <input
        type="file"
        accept=".wav,audio/wav"
        onChange={handleFile}
        style={{ marginBottom: "0.75rem" }}
      />

      <button
        disabled={!file || loading}
        onClick={submit}
        style={{ padding: "0.5rem 1rem" }}
      >
        {loading ? "Sending…" : "Check voice"}
      </button>

      {result && (
        <p style={{ marginTop: "1rem", fontWeight: "bold" }}>
          Result: {result}
        </p>
      )}
    </div>
  );
}