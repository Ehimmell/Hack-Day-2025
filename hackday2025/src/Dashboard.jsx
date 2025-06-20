import {
  Box,
  Container,
  Typography,
} from "@mui/material";
import { DatePicker } from "@mui/x-date-pickers";

// src/Dashboard.jsx

import SimpleCard from "./components/SimpleCard";
import ChartCard from "./components/ChartCard";
import CallLogTable from "./components/CallLogTable";
import HeatMapCard from "./components/HeatMapCard";
import ReviewQueueCard from "./components/ReviewQueueCard";
import ModelHealthCard from "./components/ModelHealthCard";

import "./styles/Dashboard.css";
import APIInput from "./components/APIInput";
import WavCompareCard from "./components/WavCompareCard";


const Dashboard = () => {
  const lineData = [
    { name: "Mon", human: 320, ai: 12 },
    { name: "Tue", human: 410, ai: 20 },
    { name: "Wed", human: 370, ai: 25 },
    { name: "Thu", human: 500, ai: 40 },
    { name: "Fri", human: 450, ai: 35 },
  ];

  const pieData = [
    { name: "Human Calls", value: 94 },
    { name: "AI Calls", value: 6 },
  ];

  return (
    <Container sx={{ display: "flex", flexDirection: "column", rowGap: "2rem" }}>
      <Box sx={{ display: "flex", flexDirection: { xs: "column", md: "row" }, justifyContent: "space-between", alignItems: "center" }}>
        <Box sx={{ mb: { xs: "1rem", md: "none" } }}>
          <Typography sx={{ fontSize: "2.5rem", fontWeight: 800 }}>AI Call Detection Dashboard</Typography>
          <Typography sx={{ color: "gray" }}>Welcome back, Analyst</Typography>
        </Box>
        <Box sx={{ display: "flex", flexDirection: "row", columnGap: "1rem", alignItems: "center" }}>
          <Typography sx={{ fontWeight: 600 }}>Date range:</Typography>
          <DatePicker className="date-input" label="Start" />
          <Typography sx={{ fontWeight: 600, color: "gray" }}>to</Typography>
          <DatePicker className="date-input" label="End" />
        </Box>
      </Box>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <SimpleCard header="Total Calls Analyzed" number="8432" trend="+5%" description="Compared to last week" />
        <SimpleCard header="AI-Detected Calls" number="143" trend="+12%" description="Flagged as suspicious" />
        <SimpleCard header="False Positives" number="12" trend="-2%" description="Incorrectly flagged calls" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ChartCard title="AI vs Human Calls Over Time" type="line" data={lineData} />
        <ChartCard title="Call Distribution" type="pie" data={pieData} />
      </div>

      <CallLogTable />
        <WavCompareCard />
        <APIInput />
      <HeatMapCard />
      <ReviewQueueCard />
      <ModelHealthCard />
    </Container>
  );
};

export default Dashboard;

