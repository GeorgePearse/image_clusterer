import React, { useEffect, useRef, useState } from 'react';
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import { WS_URL } from '../api';

export const LogConsole: React.FC = () => {
  const [logs, setLogs] = useState<string[]>([]);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS_URL);
      
      ws.onopen = () => {
        setLogs(prev => [...prev, "System connected."]);
      };

      ws.onmessage = (event) => {
        setLogs(prev => [...prev, event.data]);
      };

      ws.onclose = () => {
        setTimeout(connect, 2000);
      };

      wsRef.current = ws;
    };

    connect();

    return () => {
      wsRef.current?.close();
    };
  }, []);

  useEffect(() => {
    // Scroll to bottom logic
    // With ScrollArea we might need to target the viewport or just use simple div scrolling if ScrollArea is complex to control programmatically without refs to viewport.
    // Standard ScrollArea from shadcn exposes viewport via context or refs? 
    // Actually, simply putting a div at the end and scrolling it into view works if the container is scrollable.
    // But Shadcn ScrollArea wraps content in a Viewport.
    if (logsEndRef.current) {
        logsEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [logs]);

  return (
    <ScrollArea className="h-full w-full bg-card text-muted-foreground font-mono text-[10px]">
      <div className="p-4 space-y-1.5">
          {logs.map((log, i) => (
            <div key={i} className="whitespace-pre-wrap break-words border-b border-border/50 pb-1 last:border-0 hover:text-foreground transition-colors">
              <span className="text-muted-foreground/60 mr-2 select-none">[{new Date().toLocaleTimeString([], {hour12: false})}]</span>
              <span className={cn(
                  log.includes("Error") && "text-destructive",
                  log.includes("connected") && "text-green-500",
                  !log.includes("Error") && !log.includes("connected") && "text-foreground/80"
              )}>{log}</span>
            </div>
          ))}
          <div ref={logsEndRef} />
      </div>
    </ScrollArea>
  );
};
