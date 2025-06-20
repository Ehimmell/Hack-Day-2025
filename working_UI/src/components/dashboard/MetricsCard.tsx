import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface MetricsCardProps {
  title: string;
  value: string | number;
  change?: {
    value: number;
    period: string;
  };
  variant?: "default" | "success" | "warning" | "destructive" | "info";
  icon?: React.ReactNode;
  subtitle?: string;
  className?: string;
}

export function MetricsCard({
  title,
  value,
  change,
  variant = "default",
  icon,
  subtitle,
  className,
}: MetricsCardProps) {
  const getTrendIcon = (changeValue: number) => {
    if (changeValue > 0) return <TrendingUp className="h-3 w-3" />;
    if (changeValue < 0) return <TrendingDown className="h-3 w-3" />;
    return <Minus className="h-3 w-3" />;
  };

  const getTrendColor = (changeValue: number) => {
    if (changeValue > 0) return "text-success";
    if (changeValue < 0) return "text-destructive";
    return "text-muted-foreground";
  };

  const cardVariants = {
    default: "border-border",
    success: "border-success/20 bg-success/5",
    warning: "border-warning/20 bg-warning/5",
    destructive: "border-destructive/20 bg-destructive/5",
    info: "border-info/20 bg-info/5",
  };

  return (
    <Card
      className={cn(
        "transition-all duration-200 hover:shadow-metric hover:scale-[1.02]",
        cardVariants[variant],
        className,
      )}
    >
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {title}
        </CardTitle>
        {icon && (
          <div
            className={cn(
              "rounded-md p-1.5",
              variant === "success" && "bg-success/10 text-success",
              variant === "warning" && "bg-warning/10 text-warning",
              variant === "destructive" && "bg-destructive/10 text-destructive",
              variant === "info" && "bg-info/10 text-info",
              variant === "default" && "bg-primary/10 text-primary",
            )}
          >
            {icon}
          </div>
        )}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold text-foreground mb-1">
          {typeof value === "number" ? value.toLocaleString() : value}
        </div>

        {subtitle && (
          <p className="text-xs text-muted-foreground mb-2">{subtitle}</p>
        )}

        {change && (
          <div className="flex items-center space-x-1">
            <Badge
              variant="outline"
              className={cn(
                "text-xs px-1.5 py-0.5 border-0",
                getTrendColor(change.value),
                change.value > 0 && "bg-success/10",
                change.value < 0 && "bg-destructive/10",
                change.value === 0 && "bg-muted/50",
              )}
            >
              <span className="flex items-center gap-1">
                {getTrendIcon(change.value)}
                {Math.abs(change.value)}%
              </span>
            </Badge>
            <span className="text-xs text-muted-foreground">
              vs {change.period}
            </span>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
