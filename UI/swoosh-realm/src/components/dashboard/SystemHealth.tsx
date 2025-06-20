import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Activity,
  Zap,
  Clock,
  CheckCircle,
  AlertCircle,
  TrendingUp,
} from "lucide-react";
import { cn } from "@/lib/utils";

const systemMetrics = [
  {
    title: "Model Accuracy",
    value: 94.2,
    unit: "%",
    change: +1.3,
    status: "good",
    icon: <CheckCircle className="h-4 w-4" />,
  },
  {
    title: "Processing Speed",
    value: 1.2,
    unit: "sec/call",
    change: -0.2,
    status: "good",
    icon: <Zap className="h-4 w-4" />,
  },
  {
    title: "System Uptime",
    value: 99.8,
    unit: "%",
    change: 0,
    status: "excellent",
    icon: <Activity className="h-4 w-4" />,
  },
  {
    title: "Error Rate",
    value: 0.3,
    unit: "%",
    change: -0.1,
    status: "good",
    icon: <AlertCircle className="h-4 w-4" />,
  },
];

const performanceData = [
  { time: "00:00", accuracy: 94.1, precision: 92.3, recall: 91.8, f1: 92.0 },
  { time: "04:00", accuracy: 94.3, precision: 92.5, recall: 92.1, f1: 92.3 },
  { time: "08:00", accuracy: 94.0, precision: 92.1, recall: 91.9, f1: 92.0 },
  { time: "12:00", accuracy: 94.5, precision: 92.8, recall: 92.4, f1: 92.6 },
  { time: "16:00", accuracy: 94.2, precision: 92.4, recall: 92.0, f1: 92.2 },
  { time: "20:00", accuracy: 94.4, precision: 92.6, recall: 92.2, f1: 92.4 },
];

const modelVersions = [
  {
    version: "v2.4.1",
    status: "active",
    deployed: "2024-01-10",
    accuracy: 94.2,
    performance: "High",
  },
  {
    version: "v2.4.0",
    status: "staging",
    deployed: "2024-01-08",
    accuracy: 93.8,
    performance: "High",
  },
  {
    version: "v2.3.2",
    status: "deprecated",
    deployed: "2023-12-15",
    accuracy: 92.1,
    performance: "Medium",
  },
];

const getStatusColor = (status: string) => {
  switch (status) {
    case "excellent":
      return "text-success";
    case "good":
      return "text-success";
    case "warning":
      return "text-warning";
    case "error":
      return "text-destructive";
    default:
      return "text-muted-foreground";
  }
};

const getStatusBadge = (status: string) => {
  switch (status) {
    case "active":
      return "success";
    case "staging":
      return "warning";
    case "deprecated":
      return "secondary";
    default:
      return "secondary";
  }
};

const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-card border border-border rounded-md p-3 shadow-lg">
        <p className="font-medium text-foreground">{label}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} style={{ color: entry.color }} className="text-sm">
            {`${entry.dataKey}: ${entry.value}%`}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export function SystemHealth() {
  return (
    <div className="space-y-6">
      {/* System Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {systemMetrics.map((metric) => (
          <Card key={metric.title} className="relative overflow-hidden">
            <CardContent className="p-4">
              <div className="flex items-center justify-between mb-2">
                <div
                  className={cn(
                    "rounded-md p-1.5",
                    metric.status === "excellent" &&
                      "bg-success/10 text-success",
                    metric.status === "good" && "bg-success/10 text-success",
                    metric.status === "warning" && "bg-warning/10 text-warning",
                    metric.status === "error" &&
                      "bg-destructive/10 text-destructive",
                  )}
                >
                  {metric.icon}
                </div>
                {metric.change !== 0 && (
                  <Badge
                    variant="outline"
                    className={cn(
                      "text-xs px-1.5 py-0.5 border-0",
                      metric.change > 0
                        ? "bg-success/10 text-success"
                        : "bg-destructive/10 text-destructive",
                    )}
                  >
                    <TrendingUp className="h-3 w-3 mr-1" />
                    {Math.abs(metric.change)}
                    {metric.unit}
                  </Badge>
                )}
              </div>
              <div className="space-y-1">
                <h3 className="text-sm font-medium text-muted-foreground">
                  {metric.title}
                </h3>
                <div className="flex items-baseline gap-1">
                  <span
                    className={cn(
                      "text-2xl font-bold",
                      getStatusColor(metric.status),
                    )}
                  >
                    {metric.value}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {metric.unit}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Performance Trends */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="text-lg font-semibold flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Model Performance Metrics
            </CardTitle>
            <p className="text-sm text-muted-foreground">
              Real-time accuracy, precision, recall, and F1 score over the past
              24 hours
            </p>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  stroke="hsl(var(--border))"
                />
                <XAxis
                  dataKey="time"
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <YAxis
                  domain={[90, 95]}
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={12}
                />
                <Tooltip content={<CustomTooltip />} />
                <Line
                  type="monotone"
                  dataKey="accuracy"
                  stroke="hsl(var(--primary))"
                  strokeWidth={2}
                  name="Accuracy"
                />
                <Line
                  type="monotone"
                  dataKey="precision"
                  stroke="hsl(var(--success))"
                  strokeWidth={2}
                  name="Precision"
                />
                <Line
                  type="monotone"
                  dataKey="recall"
                  stroke="hsl(var(--warning))"
                  strokeWidth={2}
                  name="Recall"
                />
                <Line
                  type="monotone"
                  dataKey="f1"
                  stroke="hsl(var(--info))"
                  strokeWidth={2}
                  name="F1 Score"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Model Versions */}
        <Card>
          <CardHeader>
            <CardTitle className="text-lg font-semibold flex items-center gap-2">
              <Activity className="h-5 w-5 text-info" />
              Model Versions
            </CardTitle>
            <p className="text-sm text-muted-foreground">
              Current and previous model deployments
            </p>
          </CardHeader>
          <CardContent className="space-y-4">
            {modelVersions.map((version) => (
              <div
                key={version.version}
                className="p-3 border border-border rounded-md space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-sm font-medium">
                    {version.version}
                  </span>
                  <Badge variant={getStatusBadge(version.status)}>
                    {version.status.toUpperCase()}
                  </Badge>
                </div>

                <div className="space-y-2">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Accuracy</span>
                    <span className="font-medium">{version.accuracy}%</span>
                  </div>
                  <Progress value={version.accuracy} className="h-1.5" />
                </div>

                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>Deployed: {version.deployed}</span>
                  <span>{version.performance} Performance</span>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Processing Queue Status */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold flex items-center gap-2">
            <Clock className="h-5 w-5 text-warning" />
            Processing Queue Status
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Real-time processing performance and queue management
          </p>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Queue Length</span>
                <span className="font-medium">23 calls</span>
              </div>
              <Progress value={23} className="h-2" />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Avg Wait Time</span>
                <span className="font-medium">1.8 sec</span>
              </div>
              <Progress value={18} className="h-2" />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Throughput</span>
                <span className="font-medium">50 calls/min</span>
              </div>
              <Progress value={83} className="h-2" />
            </div>
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">CPU Usage</span>
                <span className="font-medium">67%</span>
              </div>
              <Progress value={67} className="h-2" />
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
