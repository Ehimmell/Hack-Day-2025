import React, { useState } from "react";

const APIInput = () => {
  const API = "http://localhost:5000/predict";  // change if needed

  const [file, setFile]       = useState(null);
  const [status, setStatus]   = useState("");   // idle / uploading / result
  const [label, setLabel]     = useState(null); // AI / Human

  /* handle file input */
  const handleSelect = (e) => {
    setFile(e.target.files[0] ?? null);
    setLabel(null);
    setStatus("");
  };

  /* upload */
  const send = async () => {
    if (!file) return;
    setStatus("Uploading…");
    const form = new FormData();
    form.append("file", file);

    try {
      const res  = await fetch(API, { method: "POST", body: form });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error ?? res.statusText);
      setLabel(json.label);
      setStatus("Done");
    } catch (err) {
      console.error(err);
      setLabel(null);
      setStatus("Error – see console");
    }
  };

  return (
    <div style={{ fontFamily: "sans-serif", padding: "1rem", maxWidth: 420 }}>
      <h3>Voice Checker</h3>

      <input
        type="file"
        accept=".wav,audio/wav"
        onChange={handleSelect}
      />

      {file && (
        <p style={{ margin: "0.5em 0", fontSize: 14 }}>
          Selected: <em>{file.name}</em>
        </p>
      )}

      <button
        onClick={send}
        disabled={!file || status === "Uploading…"}
        style={{ padding: "0.5em 1.5em" }}
      >
        {status === "Uploading…" ? "Sending…" : "Submit"}
      </button>

      {status && <p style={{ marginTop: "0.75em" }}>{status}</p>}
      {label && <p style={{ fontWeight: "bold" }}>Result: {label}</p>}
    </div>
  );
}

export default APIInput;