import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Badge } from "@/components/ui/badge";

const timeSeriesData = [
  { date: "Jan 8", human: 245, ai: 12, uncertain: 8 },
  { date: "Jan 9", human: 278, ai: 18, uncertain: 12 },
  { date: "Jan 10", human: 256, ai: 25, uncertain: 15 },
  { date: "Jan 11", human: 289, ai: 31, uncertain: 18 },
  { date: "Jan 12", human: 267, ai: 28, uncertain: 14 },
  { date: "Jan 13", human: 298, ai: 35, uncertain: 22 },
  { date: "Jan 14", human: 312, ai: 41, uncertain: 19 },
  { date: "Jan 15", human: 295, ai: 38, uncertain: 16 },
];

const distributionData = [
  { name: "Human", value: 2240, color: "hsl(var(--success))" },
  { name: "AI Detected", value: 227, color: "hsl(var(--destructive))" },
  { name: "Uncertain", value: 124, color: "hsl(var(--warning))" },
];

const regionData = [
  { region: "North East", aiCalls: 45, totalCalls: 486, rate: 9.3 },
  { region: "South East", aiCalls: 38, totalCalls: 521, rate: 7.3 },
  { region: "Midwest", aiCalls: 52, totalCalls: 467, rate: 11.1 },
  { region: "West Coast", aiCalls: 29, totalCalls: 398, rate: 7.3 },
  { region: "Southwest", aiCalls: 41, totalCalls: 445, rate: 9.2 },
  { region: "Northwest", aiCalls: 22, totalCalls: 274, rate: 8.0 },
];

const confidenceData = [
  { range: "0-20%", count: 8 },
  { range: "21-40%", count: 12 },
  { range: "41-60%", count: 28 },
  { range: "61-80%", count: 45 },
  { range: "81-100%", count: 134 },
];

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-card border border-border rounded-md p-3 shadow-lg">
        <p className="font-medium text-foreground">{label}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} style={{ color: entry.color }} className="text-sm">
            {`${entry.dataKey}: ${entry.value}`}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export function VisualizationCharts() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Time Series Chart */}
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle className="text-lg font-semibold">
            Call Volume Trends
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Daily breakdown of human vs AI-detected calls over the past week
          </p>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={timeSeriesData}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--border))"
              />
              <XAxis
                dataKey="date"
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
              />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              <Line
                type="monotone"
                dataKey="human"
                stroke="hsl(var(--success))"
                strokeWidth={2}
                name="Human Calls"
              />
              <Line
                type="monotone"
                dataKey="ai"
                stroke="hsl(var(--destructive))"
                strokeWidth={2}
                name="AI Detected"
              />
              <Line
                type="monotone"
                dataKey="uncertain"
                stroke="hsl(var(--warning))"
                strokeWidth={2}
                name="Uncertain"
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Distribution Pie Chart */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">
            Call Type Distribution
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Overall breakdown of call classifications
          </p>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={distributionData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) =>
                  `${name} ${(percent * 100).toFixed(1)}%`
                }
              >
                {distributionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap gap-2 mt-4">
            {distributionData.map((item) => (
              <Badge key={item.name} variant="outline" className="text-xs">
                <div
                  className="w-2 h-2 rounded-full mr-2"
                  style={{ backgroundColor: item.color }}
                />
                {item.name}: {item.value.toLocaleString()}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Regional Analysis */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold">
            Regional Hotspots
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            AI call detection rates by region
          </p>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={regionData} layout="horizontal">
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--border))"
              />
              <XAxis
                type="number"
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
              />
              <YAxis
                dataKey="region"
                type="category"
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
                width={80}
              />
              <Tooltip content={<CustomTooltip />} />
              <Bar
                dataKey="rate"
                fill="hsl(var(--primary))"
                name="AI Detection Rate (%)"
              />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Confidence Distribution */}
      <Card className="lg:col-span-2">
        <CardHeader>
          <CardTitle className="text-lg font-semibold">
            Confidence Score Distribution
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Distribution of AI detection confidence levels across all flagged
            calls
          </p>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={confidenceData}>
              <CartesianGrid
                strokeDasharray="3 3"
                stroke="hsl(var(--border))"
              />
              <XAxis
                dataKey="range"
                stroke="hsl(var(--muted-foreground))"
                fontSize={12}
              />
              <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
              <Tooltip content={<CustomTooltip />} />
              <Area
                type="monotone"
                dataKey="count"
                stroke="hsl(var(--primary))"
                fill="hsl(var(--primary))"
                fillOpacity={0.6}
                name="Number of Calls"
              />
            </AreaChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
