// src/components/ReviewQueueCard.jsx
import React from "react";
import { Card, CardContent, Typography, Button } from "@mui/material";

const ReviewQueueCard = () => {
  return (
    <Card>
      <CardContent>
        <Typography variant="h6">Calls Needing Manual Review</Typography>
        <Typography variant="body1" sx={{ my: 2 }}>
          7 calls flagged as ambiguous
        </Typography>
        <Button variant="contained">Review Now</Button>
      </CardContent>
    </Card>
  );
};

export default ReviewQueueCard;