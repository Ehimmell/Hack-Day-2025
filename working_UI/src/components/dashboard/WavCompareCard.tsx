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

const API_URL = "http://localhost:5000/classify"; // Flask endpoint

export default function WavCompareCard() {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | uploading | done | error
  const [result, setResult] = useState(null);

  const onSelect1 = (e) => {
    setFile1(e.target.files[0] ?? null);
    setResult(null);
  };
  const onSelect2 = (e) => {
    setFile2(e.target.files[0] ?? null);
    setResult(null);
  };

  const submit = async () => {
    if (!file1 || !file2) return;
    setStatus("uploading");

    const fd = new FormData();
    fd.append("audio1", file1);
    fd.append("audio2", file2);

    try {
      const resp = await fetch(API_URL, {
        method: "POST",
        body: fd,
        headers: {
          /** extra header requested */
          enctype: "multipart/form-data",
        },
      });
      const jsn = await resp.json();
      if (!resp.ok) throw new Error(jsn.error ?? resp.statusText);
      setResult(jsn); // whatever structure the server returns
      setStatus("done");
    } catch (err) {
      console.error(err);
      setStatus("error");
    }
  };

  const uploading = status === "uploading";

  return (
    <Card sx={{ p: 2 }}>
      <CardContent>
        {/* header row */}
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              Voice Comparison
            </Typography>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              {result ? JSON.stringify(result) : "Choose two WAV files"}
            </Typography>
            {status === "error" && (
              <Typography variant="body2" color="error.main">
                Upload failed – check console
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

        {/* action buttons */}
        <Box sx={{ mt: 3, display: "flex", gap: 1, flexWrap: "wrap" }}>
          <Button
            component="label"
            variant="outlined"
            startIcon={<UploadFileIcon />}
            disabled={uploading}
          >
            Choose WAV 1
            <input hidden type="file" accept=".wav,audio/wav" onChange={onSelect1} />
          </Button>

          <Button
            component="label"
            variant="outlined"
            startIcon={<UploadFileIcon />}
            disabled={uploading}
          >
            Choose WAV 2
            <input hidden type="file" accept=".wav,audio/wav" onChange={onSelect2} />
          </Button>

          <Button
            variant="contained"
            onClick={submit}
            disabled={!file1 || !file2 || uploading}
          >
            {uploading ? "Sending…" : "Compare"}
          </Button>
        </Box>

        {/* filenames */}
        {(file1 || file2) && (
          <Typography variant="caption" sx={{ mt: 1, display: "block" }}>
            {file1 && <>File 1: {file1.name}<br /></>}
            {file2 && <>File 2: {file2.name}</>}
          </Typography>
        )}

        {uploading && <LinearProgress sx={{ mt: 2 }} />}
      </CardContent>
    </Card>
  );
}