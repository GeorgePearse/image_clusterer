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
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs, isOpen]);

  return (
    <div className={`bg-gray-900 text-green-400 font-mono text-xs transition-all duration-300 border-r border-gray-700 h-full flex flex-col flex-shrink-0 ${isOpen ? 'w-80' : 'w-12'}`}>
      <div 
        className="flex items-center justify-between p-3 h-12 bg-gray-800 cursor-pointer hover:bg-gray-700 border-b border-gray-700"
        onClick={() => setIsOpen(!isOpen)}
        title={isOpen ? "Collapse logs" : "Expand logs"}
      >
        <div className="flex items-center gap-2 overflow-hidden">
          <Terminal size={16} className="flex-shrink-0" />
          {isOpen && (
            <>
              <span className="font-semibold whitespace-nowrap">System Logs</span>
              <span className="bg-gray-700 text-gray-300 px-2 py-0.5 rounded text-[10px] whitespace-nowrap">{logs.length}</span>
            </>
          )}
        </div>
        {isOpen ? <ChevronLeft size={16} /> : <ChevronRight size={16} />}
      </div>
      
      {isOpen && (
        <div className="p-4 flex-1 overflow-y-auto overflow-x-hidden">
          {logs.map((log, i) => (
            <div key={i} className="mb-1 whitespace-pre-wrap font-mono opacity-90 border-b border-gray-800 pb-1 break-words">
              {log}
            </div>
          ))}
          <div ref={logsEndRef} />
        </div>
      )}
      
      {!isOpen && (
         <div className="flex flex-col items-center mt-4 gap-2 opacity-50">
             <span className="text-[10px] writing-vertical-rl rotate-180 select-none">Logs ({logs.length})</span>
         </div>
      )}
    </div>
  );
};
