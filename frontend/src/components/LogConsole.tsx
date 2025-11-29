import React, { useEffect, useRef, useState } from 'react';
import { Terminal, ChevronLeft, ChevronRight } from 'lucide-react';
import { WS_URL } from '../api';

export const LogConsole: React.FC = () => {
  const [logs, setLogs] = useState<string[]>([]);
  const [isOpen, setIsOpen] = useState(true);
  const logsEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS_URL);
      
      ws.onopen = () => {
        setLogs(prev => [...prev, "--- System Logs Connected ---"]);
      };

      ws.onmessage = (event) => {
        setLogs(prev => [...prev, event.data]);
      };

      ws.onclose = () => {
        // Try reconnect in 2s
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
    if (isOpen && logsEndRef.current) {
      // Use scrollTop instead of scrollIntoView to prevent layout jitter
      const parent = logsEndRef.current.parentElement;
      if (parent) {
        parent.scrollTop = parent.scrollHeight;
      }
    }
  }, [logs, isOpen]);

  return (
    <div className="bg-[#111] text-green-400 font-mono text-xs h-full flex flex-col border-l border-[#333]">
      <div 
        className="flex items-center justify-between p-3 h-10 bg-[#1a1a1a] border-b border-[#333]"
      >
        <div className="flex items-center gap-2 overflow-hidden">
          <Terminal size={14} className="flex-shrink-0 text-gray-400" />
          <span className="font-semibold whitespace-nowrap text-gray-300">System Logs</span>
          <span className="bg-[#333] text-gray-300 px-1.5 rounded text-[10px] min-w-[20px] text-center">{logs.length}</span>
        </div>
      </div>
      
      <div className="p-3 flex-1 overflow-y-auto overflow-x-hidden scrollbar-thin scrollbar-thumb-gray-700 scrollbar-track-transparent">
          {logs.map((log, i) => (
            <div key={i} className="mb-1.5 whitespace-pre-wrap font-mono opacity-80 border-b border-gray-800/50 pb-1 break-words leading-relaxed text-[11px]">
              <span className="text-gray-500 mr-2">[{new Date().toLocaleTimeString()}]</span>
              {log}
            </div>
          ))}
          <div ref={logsEndRef} />
      </div>
    </div>
  );
};
