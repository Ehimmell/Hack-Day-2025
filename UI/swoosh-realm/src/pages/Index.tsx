import { MetricsCard } from "@/components/dashboard/MetricsCard";
import { CallInsightsTable } from "@/components/dashboard/CallInsightsTable";
import { VisualizationCharts } from "@/components/dashboard/VisualizationCharts";
import { FraudMonitoring } from "@/components/dashboard/FraudMonitoring";
import { SystemHealth } from "@/components/dashboard/SystemHealth";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Phone,
  Brain,
  Users,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Activity,
  Settings,
  Download,
  RefreshCw,
  Calendar,
  ChevronDown,
} from "lucide-react";

const Index = () => {
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <div className="border-b border-border bg-card">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-3">
                <div className="bg-primary/10 p-2 rounded-lg">
                  <Brain className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold text-foreground">
                    SecureCall Analytics
                  </h1>
                  <p className="text-sm text-muted-foreground">
                    AI Call Detection & Fraud Prevention Dashboard
                  </p>
                </div>
              </div>
              <Badge
                variant="outline"
                className="bg-success/10 text-success border-success/20"
              >
                Live
              </Badge>
            </div>

            <div className="flex items-center gap-3">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline" className="gap-2">
                    <Calendar className="h-4 w-4" />
                    Last 7 days
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuLabel>Time Range</DropdownMenuLabel>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem>Last 24 hours</DropdownMenuItem>
                  <DropdownMenuItem>Last 7 days</DropdownMenuItem>
                  <DropdownMenuItem>Last 30 days</DropdownMenuItem>
                  <DropdownMenuItem>Custom range</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>

              <Button variant="outline" size="sm" className="gap-2">
                <Download className="h-4 w-4" />
                Export
              </Button>

              <Button variant="outline" size="sm" className="gap-2">
                <RefreshCw className="h-4 w-4" />
                Refresh
              </Button>

              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="container mx-auto px-6 py-6">
        <Tabs defaultValue="overview" className="space-y-6">
          <TabsList className="grid w-full lg:w-auto lg:inline-flex grid-cols-5 lg:grid-cols-none">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
            <TabsTrigger value="fraud">Fraud & Compliance</TabsTrigger>
            <TabsTrigger value="system">System Health</TabsTrigger>
            <TabsTrigger value="settings">Settings</TabsTrigger>
          </TabsList>

          {/* Overview Tab */}
          <TabsContent value="overview" className="space-y-6">
            {/* KPI Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricsCard
                title="Total Calls Analyzed"
                value="2,591"
                change={{ value: 12.3, period: "last week" }}
                icon={<Phone className="h-4 w-4" />}
                subtitle="All processed calls in selected period"
              />
              <MetricsCard
                title="AI-Detected Calls"
                value="227"
                change={{ value: -8.1, period: "last week" }}
                variant="destructive"
                icon={<Brain className="h-4 w-4" />}
                subtitle="8.8% of total calls flagged"
              />
              <MetricsCard
                title="Human Calls"
                value="2,240"
                change={{ value: 14.2, period: "last week" }}
                variant="success"
                icon={<Users className="h-4 w-4" />}
                subtitle="86.5% confirmed human callers"
              />
              <MetricsCard
                title="Uncertain Cases"
                value="124"
                change={{ value: 23.5, period: "last week" }}
                variant="warning"
                icon={<AlertTriangle className="h-4 w-4" />}
                subtitle="4.8% requiring manual review"
              />
            </div>

            {/* Model Performance Indicators */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <MetricsCard
                title="Model Accuracy"
                value="94.2%"
                change={{ value: 1.3, period: "last month" }}
                variant="success"
                icon={<TrendingUp className="h-4 w-4" />}
                subtitle="Overall detection accuracy"
              />
              <MetricsCard
                title="False Positive Rate"
                value="2.1%"
                change={{ value: -0.4, period: "last month" }}
                variant="info"
                icon={<TrendingDown className="h-4 w-4" />}
                subtitle="Incorrectly flagged human calls"
              />
              <MetricsCard
                title="System Uptime"
                value="99.8%"
                change={{ value: 0, period: "last month" }}
                variant="success"
                icon={<Activity className="h-4 w-4" />}
                subtitle="Service availability"
              />
            </div>

            {/* Call Insights Table */}
            <CallInsightsTable />
          </TabsContent>

          {/* Analytics Tab */}
          <TabsContent value="analytics" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricsCard
                title="Daily AI Detection Rate"
                value="8.8%"
                change={{ value: -2.1, period: "yesterday" }}
                variant="warning"
                icon={<Brain className="h-4 w-4" />}
              />
              <MetricsCard
                title="Peak Detection Time"
                value="2:30 PM"
                subtitle="Highest AI call frequency"
                icon={<AlertTriangle className="h-4 w-4" />}
              />
              <MetricsCard
                title="Average Call Duration"
                value="3:42"
                change={{ value: 8.2, period: "last week" }}
                subtitle="AI calls vs 4:18 for humans"
                icon={<Phone className="h-4 w-4" />}
              />
              <MetricsCard
                title="Confidence Distribution"
                value="78.3%"
                subtitle="Avg confidence for flagged calls"
                variant="info"
                icon={<TrendingUp className="h-4 w-4" />}
              />
            </div>

            <VisualizationCharts />
          </TabsContent>

          {/* Fraud & Compliance Tab */}
          <TabsContent value="fraud" className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricsCard
                title="Active Alerts"
                value="4"
                variant="destructive"
                icon={<AlertTriangle className="h-4 w-4" />}
                subtitle="Requiring immediate attention"
              />
              <MetricsCard
                title="Repeat Offenders"
                value="12"
                change={{ value: 50, period: "last week" }}
                variant="warning"
                icon={<Users className="h-4 w-4" />}
                subtitle="Frequent AI callers identified"
              />
              <MetricsCard
                title="Compliance Violations"
                value="3"
                variant="destructive"
                icon={<AlertTriangle className="h-4 w-4" />}
                subtitle="Regulatory violations detected"
              />
              <MetricsCard
                title="Manual Reviews Pending"
                value="23"
                change={{ value: -12, period: "yesterday" }}
                variant="info"
                icon={<Activity className="h-4 w-4" />}
                subtitle="Calls awaiting human review"
              />
            </div>

            <FraudMonitoring />
          </TabsContent>

          {/* System Health Tab */}
          <TabsContent value="system" className="space-y-6">
            <SystemHealth />
          </TabsContent>

          {/* Settings Tab */}
          <TabsContent value="settings" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="bg-card border border-border rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-4">
                  Detection Thresholds
                </h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Configure AI detection sensitivity and confidence thresholds
                </p>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">
                      AI Detection Threshold
                    </label>
                    <div className="text-sm text-muted-foreground">
                      75% confidence
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">
                      Auto-Flag Threshold
                    </label>
                    <div className="text-sm text-muted-foreground">
                      85% confidence
                    </div>
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">
                      Manual Review Trigger
                    </label>
                    <div className="text-sm text-muted-foreground">
                      60-85% confidence range
                    </div>
                  </div>
                </div>
                <Button className="mt-4">Update Thresholds</Button>
              </div>

              <div className="bg-card border border-border rounded-lg p-6">
                <h3 className="text-lg font-semibold mb-4">User Management</h3>
                <p className="text-sm text-muted-foreground mb-4">
                  Manage user roles and access permissions
                </p>
                <div className="space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Claims Analysts</span>
                    <Badge variant="outline">Read Only</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Fraud Detection Team</span>
                    <Badge variant="secondary">Full Access</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">Compliance Officers</span>
                    <Badge variant="secondary">Full Access</Badge>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">IT Administrators</span>
                    <Badge variant="outline">Admin</Badge>
                  </div>
                </div>
                <Button className="mt-4">Manage Users</Button>
              </div>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Index;
