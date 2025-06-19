// src/components/ModelHealthCard.jsx
import React from "react";
import { Card, CardContent, Typography, LinearProgress, Box } from "@mui/material";

const ModelHealthCard = () => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6">Model Health</Typography>
        <Box sx={{ my: 2 }}>
          <Typography variant="body2">Precision</Typography>
          <LinearProgress variant="determinate" value={92} />
        </Box>
        <Box sx={{ my: 2 }}>
          <Typography variant="body2">Recall</Typography>
          <LinearProgress variant="determinate" value={88} />
        </Box>
      </CardContent>
    </Card>
  );
};

export default ModelHealthCard;
