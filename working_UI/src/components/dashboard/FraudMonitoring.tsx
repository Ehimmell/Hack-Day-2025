import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  AlertTriangle,
  Phone,
  MapPin,
  Clock,
  Eye,
  MoreHorizontal,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface FraudAlert {
  id: string;
  type: "repeat-offender" | "hotspot" | "regulatory" | "anomaly";
  severity: "high" | "medium" | "low";
  title: string;
  description: string;
  location?: string;
  timestamp: string;
  callCount?: number;
  phoneNumber?: string;
}

const fraudAlerts: FraudAlert[] = [
  {
    id: "ALERT-001",
    type: "repeat-offender",
    severity: "high",
    title: "Frequent AI Caller Detected",
    description:
      "Phone number ****-****-1234 has been flagged for AI calls 8 times in the past 24 hours",
    phoneNumber: "****-****-1234",
    timestamp: "2 hours ago",
    callCount: 8,
  },
  {
    id: "ALERT-002",
    type: "hotspot",
    severity: "medium",
    title: "Regional Anomaly Detected",
    description: "Midwest region showing 40% increase in AI call detection",
    location: "Midwest Region",
    timestamp: "4 hours ago",
    callCount: 23,
  },
  {
    id: "ALERT-003",
    type: "regulatory",
    severity: "high",
    title: "Consent Violation",
    description:
      "AI caller failed to disclose artificial nature during call recording",
    timestamp: "6 hours ago",
    callCount: 1,
  },
  {
    id: "ALERT-004",
    type: "anomaly",
    severity: "low",
    title: "Unusual Call Pattern",
    description: "Detected suspicious timing pattern in outbound calls",
    timestamp: "1 day ago",
    callCount: 12,
  },
];

const pendingReviews = [
  {
    id: "CALL-234",
    callerId: "****-****-9876",
    confidence: 67,
    flaggedReason: "Low confidence score",
    timestamp: "15 mins ago",
    priority: "medium",
  },
  {
    id: "CALL-235",
    callerId: "****-****-5432",
    confidence: 45,
    flaggedReason: "Unusual voice patterns",
    timestamp: "28 mins ago",
    priority: "high",
  },
  {
    id: "CALL-236",
    callerId: "****-****-1098",
    confidence: 72,
    flaggedReason: "Repeat caller",
    timestamp: "1 hour ago",
    priority: "low",
  },
];

const getSeverityColor = (severity: string) => {
  switch (severity) {
    case "high":
      return "destructive";
    case "medium":
      return "warning";
    case "low":
      return "info";
    default:
      return "secondary";
  }
};

const getTypeIcon = (type: string) => {
  switch (type) {
    case "repeat-offender":
      return <Phone className="h-4 w-4" />;
    case "hotspot":
      return <MapPin className="h-4 w-4" />;
    case "regulatory":
      return <AlertTriangle className="h-4 w-4" />;
    case "anomaly":
      return <Clock className="h-4 w-4" />;
    default:
      return <AlertTriangle className="h-4 w-4" />;
  }
};

const getPriorityColor = (priority: string) => {
  switch (priority) {
    case "high":
      return "text-destructive";
    case "medium":
      return "text-warning";
    case "low":
      return "text-info";
    default:
      return "text-muted-foreground";
  }
};

export function FraudMonitoring() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      {/* Fraud Alerts */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold flex items-center gap-2">
            <AlertTriangle className="h-5 w-5 text-warning" />
            Fraud Alerts
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Real-time fraud detection and compliance monitoring
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          {fraudAlerts.map((alert) => (
            <Alert
              key={alert.id}
              className={cn(
                "border-l-4",
                alert.severity === "high" &&
                  "border-l-destructive bg-destructive/5",
                alert.severity === "medium" && "border-l-warning bg-warning/5",
                alert.severity === "low" && "border-l-info bg-info/5",
              )}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 flex-1">
                  <div
                    className={cn(
                      "rounded-md p-1.5",
                      alert.severity === "high" &&
                        "bg-destructive/10 text-destructive",
                      alert.severity === "medium" &&
                        "bg-warning/10 text-warning",
                      alert.severity === "low" && "bg-info/10 text-info",
                    )}
                  >
                    {getTypeIcon(alert.type)}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="text-sm font-semibold text-foreground">
                        {alert.title}
                      </h4>
                      <Badge
                        variant={getSeverityColor(alert.severity)}
                        className="text-xs"
                      >
                        {alert.severity.toUpperCase()}
                      </Badge>
                    </div>
                    <AlertDescription className="text-xs text-muted-foreground mb-2">
                      {alert.description}
                    </AlertDescription>
                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      {alert.location && (
                        <span className="flex items-center gap-1">
                          <MapPin className="h-3 w-3" />
                          {alert.location}
                        </span>
                      )}
                      {alert.phoneNumber && (
                        <span className="flex items-center gap-1">
                          <Phone className="h-3 w-3" />
                          {alert.phoneNumber}
                        </span>
                      )}
                      {alert.callCount && <span>{alert.callCount} calls</span>}
                      <span className="flex items-center gap-1">
                        <Clock className="h-3 w-3" />
                        {alert.timestamp}
                      </span>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-1">
                  <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                    <Eye className="h-3 w-3" />
                  </Button>
                  <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                    <MoreHorizontal className="h-3 w-3" />
                  </Button>
                </div>
              </div>
            </Alert>
          ))}

          <div className="pt-2">
            <Button variant="outline" className="w-full">
              View All Alerts
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Manual Review Queue */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg font-semibold flex items-center gap-2">
            <Eye className="h-5 w-5 text-info" />
            Manual Review Queue
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Calls pending human review due to low confidence or anomalies
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          {pendingReviews.map((review) => (
            <div
              key={review.id}
              className="flex items-center justify-between p-3 border border-border rounded-md hover:bg-muted/50 transition-colors"
            >
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className="font-mono text-sm font-medium">
                    {review.id}
                  </span>
                  <Badge variant="outline" className="text-xs">
                    {review.callerId}
                  </Badge>
                </div>
                <p className="text-xs text-muted-foreground mb-2">
                  {review.flaggedReason}
                </p>
                <div className="flex items-center gap-4 text-xs">
                  <span
                    className={cn(
                      "font-semibold",
                      review.confidence >= 70
                        ? "text-warning"
                        : "text-destructive",
                    )}
                  >
                    {review.confidence}% confidence
                  </span>
                  <span
                    className={cn(
                      "font-medium",
                      getPriorityColor(review.priority),
                    )}
                  >
                    {review.priority.toUpperCase()} priority
                  </span>
                  <span className="text-muted-foreground">
                    {review.timestamp}
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-1">
                <Button variant="outline" size="sm" className="text-xs px-2">
                  Review
                </Button>
                <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                  <MoreHorizontal className="h-3 w-3" />
                </Button>
              </div>
            </div>
          ))}

          <div className="pt-2">
            <Button variant="outline" className="w-full">
              View Full Queue ({pendingReviews.length + 8} pending)
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
