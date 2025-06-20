// src/components/HeatMapCard.jsx
import React from "react";
import { Card, CardContent, Typography } from "@mui/material";

const HeatMapCard = () => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6">AI Detection Heatmap</Typography>
        <div style={{ height: "200px", backgroundColor: "#f0f0f0", display: "flex", alignItems: "center", justifyContent: "center" }}>
          <Typography variant="body2" color="text.secondary">
            [Heatmap visualization placeholder]
          </Typography>
        </div>
      </CardContent>
    </Card>
  );
};

export default HeatMapCard;
