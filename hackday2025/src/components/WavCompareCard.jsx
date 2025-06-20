// src/components/WavCompareCard.jsx
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
import CompareIcon from "@mui/icons-material/CompareArrows";

/* Change if your Flask server runs elsewhere */
const API_URL = "http://localhost:5000/classify";

const WavCompareCard = () => {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [status, setStatus] = useState("idle"); // idle · uploading · done · error
  const [result, setResult] = useState(null);   // whatever JSON returns

  /* choose file handlers */
  const choose1 = e => { setFile1(e.target.files[0] ?? null); setResult(null); };
  const choose2 = e => { setFile2(e.target.files[0] ?? null); setResult(null); };

  /* upload */
  const submit = async () => {
    if (!file1 || !file2) return;
    setStatus("uploading");

    const form = new FormData();
    form.append("audio1", file1);
    form.append("audio2", file2);

    try {
      const r   = await fetch(API_URL, { method: "POST", body: form });
      const jsn = await r.json();
      if (!r.ok) throw new Error(jsn.error ?? r.statusText);
      setResult(jsn);          // e.g. { "sameSpeaker": true, ... }
      setStatus("done");
    } catch (err) {
      console.error(err);
      setStatus("error");
    }
  };

  const isUploading = status === "uploading";

  return (
    <Card sx={{ p: 2 }}>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Voice Comparison
            </Typography>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              {result ? JSON.stringify(result) : "Upload two WAVs"}
            </Typography>
            {status === "error" && (
              <Typography variant="body2" color="error.main">
                Server error – see console
              </Typography>
            )}
          </Box>

          <Avatar
            variant="rounded"
            sx={{ width: 56, height: 56, bgcolor: "primary.light" }}
          >
            <CompareIcon fontSize="large" />
          </Avatar>
        </Box>

        {/* buttons */}
        <Box sx={{ mt: 3, display: "flex", gap: 1, flexWrap: "wrap" }}>
          <Button
            component="label"
            variant="outlined"
            startIcon={<UploadFileIcon />}
            disabled={isUploading}
          >
            Choose WAV 1
            <input hidden type="file" accept=".wav,audio/wav" onChange={choose1} />
          </Button>

          <Button
            component="label"
            variant="outlined"
            startIcon={<UploadFileIcon />}
            disabled={isUploading}
          >
            Choose WAV 2
            <input hidden type="file" accept=".wav,audio/wav" onChange={choose2} />
          </Button>

          <Button
            variant="contained"
            onClick={submit}
            disabled={!file1 || !file2 || isUploading}
          >
            {isUploading ? "Sending…" : "Compare"}
          </Button>
        </Box>

        {/* filenames */}
        {(file1 || file2) && (
          <Typography
            variant="caption"
            color="text.secondary"
            component="div"
            sx={{ mt: 1 }}
          >
            {file1 && <>File 1: {file1.name}<br /></>}
            {file2 && <>File 2: {file2.name}</>}
          </Typography>
        )}

        {isUploading && <LinearProgress sx={{ mt: 2 }} />}
      </CardContent>
    </Card>
  );
}

export default WavCompareCard;