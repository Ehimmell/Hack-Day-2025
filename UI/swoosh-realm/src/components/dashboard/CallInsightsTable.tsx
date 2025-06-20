import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Search,
  Filter,
  Play,
  MoreHorizontal,
  Eye,
  AlertTriangle,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface CallRecord {
  id: string;
  callerId: string;
  timestamp: string;
  duration: string;
  channel: "inbound" | "outbound";
  detectedType: "human" | "ai" | "uncertain";
  confidence: number;
  flagged: boolean;
  reviewed: boolean;
  reviewerOutcome?: "confirmed-ai" | "false-alarm" | "pending";
}

const mockCallData: CallRecord[] = [
  {
    id: "CALL-001",
    callerId: "****-****-1234",
    timestamp: "2024-01-15 14:32:00",
    duration: "04:23",
    channel: "inbound",
    detectedType: "ai",
    confidence: 94,
    flagged: true,
    reviewed: true,
    reviewerOutcome: "confirmed-ai",
  },
  {
    id: "CALL-002",
    callerId: "****-****-5678",
    timestamp: "2024-01-15 14:28:00",
    duration: "02:15",
    channel: "outbound",
    detectedType: "human",
    confidence: 87,
    flagged: false,
    reviewed: false,
  },
  {
    id: "CALL-003",
    callerId: "****-****-9012",
    timestamp: "2024-01-15 14:25:00",
    duration: "06:41",
    channel: "inbound",
    detectedType: "uncertain",
    confidence: 63,
    flagged: true,
    reviewed: false,
  },
  {
    id: "CALL-004",
    callerId: "****-****-3456",
    timestamp: "2024-01-15 14:20:00",
    duration: "03:18",
    channel: "inbound",
    detectedType: "ai",
    confidence: 89,
    flagged: true,
    reviewed: true,
    reviewerOutcome: "false-alarm",
  },
  {
    id: "CALL-005",
    callerId: "****-****-7890",
    timestamp: "2024-01-15 14:15:00",
    duration: "01:52",
    channel: "outbound",
    detectedType: "human",
    confidence: 92,
    flagged: false,
    reviewed: false,
  },
];

export function CallInsightsTable() {
  const [searchTerm, setSearchTerm] = useState("");
  const [filterType, setFilterType] = useState<string>("all");
  const [filterConfidence, setFilterConfidence] = useState<string>("all");

  const getTypeColor = (type: string) => {
    switch (type) {
      case "ai":
        return "destructive";
      case "human":
        return "success";
      case "uncertain":
        return "warning";
      default:
        return "secondary";
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 85) return "text-success";
    if (confidence >= 70) return "text-warning";
    return "text-destructive";
  };

  const getReviewOutcomeColor = (outcome: string) => {
    switch (outcome) {
      case "confirmed-ai":
        return "destructive";
      case "false-alarm":
        return "success";
      default:
        return "warning";
    }
  };

  const filteredData = mockCallData.filter((call) => {
    const matchesSearch =
      call.callerId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      call.id.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesType =
      filterType === "all" || call.detectedType === filterType;
    const matchesConfidence =
      filterConfidence === "all" ||
      (filterConfidence === "high" && call.confidence >= 85) ||
      (filterConfidence === "medium" &&
        call.confidence >= 70 &&
        call.confidence < 85) ||
      (filterConfidence === "low" && call.confidence < 70);

    return matchesSearch && matchesType && matchesConfidence;
  });

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg font-semibold">Call Insights</CardTitle>
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search calls..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-8 w-64"
              />
            </div>
            <Select value={filterType} onValueChange={setFilterType}>
              <SelectTrigger className="w-32">
                <SelectValue placeholder="Type" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Types</SelectItem>
                <SelectItem value="ai">AI</SelectItem>
                <SelectItem value="human">Human</SelectItem>
                <SelectItem value="uncertain">Uncertain</SelectItem>
              </SelectContent>
            </Select>
            <Select
              value={filterConfidence}
              onValueChange={setFilterConfidence}
            >
              <SelectTrigger className="w-36">
                <SelectValue placeholder="Confidence" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Confidence</SelectItem>
                <SelectItem value="high">High (85%+)</SelectItem>
                <SelectItem value="medium">Medium (70-84%)</SelectItem>
                <SelectItem value="low">Low (&lt;70%)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div className="rounded-md border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Call ID</TableHead>
                <TableHead>Caller ID</TableHead>
                <TableHead>Timestamp</TableHead>
                <TableHead>Duration</TableHead>
                <TableHead>Channel</TableHead>
                <TableHead>Detected Type</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Review</TableHead>
                <TableHead className="w-12"></TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredData.length === 0 ? (
                <TableRow>
                  <TableCell
                    colSpan={10}
                    className="text-center py-8 text-muted-foreground"
                  >
                    No calls found matching your criteria
                  </TableCell>
                </TableRow>
              ) : (
                filteredData.map((call) => (
                  <TableRow key={call.id} className="hover:bg-muted/50">
                    <TableCell className="font-mono text-sm">
                      {call.id}
                    </TableCell>
                    <TableCell className="font-mono text-sm">
                      {call.callerId}
                    </TableCell>
                    <TableCell className="text-sm">{call.timestamp}</TableCell>
                    <TableCell className="text-sm">{call.duration}</TableCell>
                    <TableCell>
                      <Badge variant="outline" className="text-xs">
                        {call.channel}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <Badge variant={getTypeColor(call.detectedType)}>
                        {call.detectedType.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell>
                      <span
                        className={cn(
                          "font-semibold",
                          getConfidenceColor(call.confidence),
                        )}
                      >
                        {call.confidence}%
                      </span>
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        {call.flagged && (
                          <AlertTriangle className="h-4 w-4 text-warning" />
                        )}
                        <Badge
                          variant={call.flagged ? "destructive" : "secondary"}
                          className="text-xs"
                        >
                          {call.flagged ? "Flagged" : "Clear"}
                        </Badge>
                      </div>
                    </TableCell>
                    <TableCell>
                      {call.reviewed ? (
                        <Badge
                          variant={getReviewOutcomeColor(
                            call.reviewerOutcome || "pending",
                          )}
                        >
                          {call.reviewerOutcome
                            ?.replace("-", " ")
                            .toUpperCase() || "REVIEWED"}
                        </Badge>
                      ) : (
                        <Badge variant="outline">Pending</Badge>
                      )}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-1">
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0"
                        >
                          <Play className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0"
                        >
                          <Eye className="h-3 w-3" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          className="h-8 w-8 p-0"
                        >
                          <MoreHorizontal className="h-3 w-3" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>

        <div className="flex items-center justify-between mt-4 text-sm text-muted-foreground">
          <span>
            Showing {filteredData.length} of {mockCallData.length} calls
          </span>
          <div className="flex items-center gap-2">
            <Button variant="outline" size="sm" disabled>
              Previous
            </Button>
            <Button variant="outline" size="sm" disabled>
              Next
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
