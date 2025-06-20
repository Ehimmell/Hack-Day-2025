// src/components/WavUploadCard.jsx
import React, { useState } from "react";
import {
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Box,
  Avatar,
} from "@mui/material";
import UploadFileIcon from "@mui/icons-material/UploadFile";
import GraphicEqIcon from "@mui/icons-material/GraphicEq";

/* -------------------------------------------------- */
/*                CONFIGURE YOUR API URL              */
/* -------------------------------------------------- */
const API_URL = "http://localhost:5000/predict";

export default function WavUploadCard() {
  const [file, setFile]       = useState(null);
  const [status, setStatus]   = useState("idle");   // idle | uploading | done | error
  const [label, setLabel]     = useState(null);     // AI | Human

  const onSelect = (e) => {
    setFile(e.target.files[0] ?? null);
    setStatus("idle");
    setLabel(null);
  };

  const onSubmit = async () => {
    if (!file) return;
    setStatus("uploading");

    const body = new FormData();
    body.append("file", file);

    try {
      const res  = await fetch(API_URL, { method: "POST", body });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error ?? res.statusText);
      setLabel(json.label);
      setStatus("done");
    } catch (err) {
      console.error(err);
      setStatus("error");
    }
  };

  /* ---- convenience flags ---- */
  const uploading = status === "uploading";
  const success   = status === "done";
  const failed    = status === "error";

  return (
    <Card sx={{ p: 2 }}>
      <CardContent>
        {/* header row */}
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Voice Classification
            </Typography>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              {label ? `Result: ${label}` : "Upload a WAV"}
            </Typography>
            {success && (
              <Typography
                variant="body2"
                color={label === "AI" ? "error.main" : "success.main"}
              >
                {label === "AI" ? "Likely synthetic" : "Likely human"}
              </Typography>
            )}
            {failed && (
              <Typography variant="body2" color="error.main">
                Server error – see console
              </Typography>
            )}
          </Box>

          {/* icon / avatar */}
          <Avatar
            variant="rounded"
            sx={{ width: 56, height: 56, bgcolor: "primary.light" }}
          >
            <GraphicEqIcon fontSize="large" />
          </Avatar>
        </Box>

        {/* file selector & button */}
        <Box sx={{ mt: 3, display: "flex", gap: 1 }}>
          <Button
            component="label"
            variant="outlined"
            startIcon={<UploadFileIcon />}
            disabled={uploading}
          >
            Choose WAV
            <input
              hidden
              type="file"
              accept=".wav,audio/wav"
              onChange={onSelect}
            />
          </Button>

          <Button
            variant="contained"
            onClick={onSubmit}
            disabled={!file || uploading}
          >
            {uploading ? "Sending…" : "Submit"}
          </Button>
        </Box>

        {/* show filename */}
        {file && (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: "block" }}>
            Selected: {file.name}
          </Typography>
        )}

        {/* progress bar */}
        {uploading && <LinearProgress sx={{ mt: 2 }} />}
      </CardContent>
    </Card>
  );
}