// src/components/CallLogTable.jsx
import React from "react";
import { Table, TableBody, TableCell, TableContainer, TableHead, TableRow, Paper, Typography } from "@mui/material";

const sampleLogs = [
  { time: "10:32 AM", caller: "+1 555-123-4567", type: "AI" },
  { time: "11:05 AM", caller: "+1 555-987-6543", type: "Human" },
  { time: "1:20 PM", caller: "+1 555-222-3333", type: "AI" },
];

const CallLogTable = () => {
  return (
    <Paper>
      <Typography variant="h6" sx={{ p: 2 }}>
        Call Log Summary
      </Typography>
      <TableContainer>
        <Table>
          <TableHead>
            <TableRow>
              <TableCell>Time</TableCell>
              <TableCell>Caller</TableCell>
              <TableCell>Detected Type</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {sampleLogs.map((log, i) => (
              <TableRow key={i}>
                <TableCell>{log.time}</TableCell>
                <TableCell>{log.caller}</TableCell>
                <TableCell>{log.type}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </TableContainer>
    </Paper>
  );
};

export default CallLogTable;