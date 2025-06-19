// src/components/SimpleCard.jsx
import React from "react";
import { Card, CardContent, Typography, Box, Avatar } from "@mui/material";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";

const SimpleCard = ({ img, header, mobileHeader, number, trend, description }) => {
  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="subtitle2" color="text.secondary">
              {mobileHeader || header}
            </Typography>
            <Typography variant="h5" sx={{ fontWeight: 600 }}>
              {number}
            </Typography>
            <Typography variant="body2" color="green">
              <TrendingUpIcon fontSize="small" /> {trend}
            </Typography>
          </Box>
          <Avatar variant="rounded" src={img} sx={{ width: 56, height: 56 }} />
        </Box>
        {description && (
          <Typography variant="body2" color="text.secondary" mt={2}>
            {description}
          </Typography>
        )}
      </CardContent>
    </Card>
  );
};

export default SimpleCard;