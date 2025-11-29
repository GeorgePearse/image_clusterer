import { useEffect, useState, useRef, useCallback } from 'react';
import './App.css';
import { LogConsole } from './components/LogConsole';
import { ScatterPlot } from './components/ScatterPlot';
import { fetchNextSample, sendLabel, fetchPoints, fetchStatus, fetchKnnConfig, setKnnConfig, type NextSampleResponse, type Point, type StatusResponse, type KnnConfig } from './api';
import { Check, ChevronRight, Layout, Terminal, Sparkles, ArrowLeft, Search, Loader2, GripVertical, Settings2 } from 'lucide-react';
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

const CLASSES = [
  "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
];

// Resize Handle Component
function ResizeHandle({ onResize, side }: { onResize: (delta: number) => void; side: 'left' | 'right' }) {
  const [isDragging, setIsDragging] = useState(false);
  const startXRef = useRef(0);

  const handleMouseDown = (e: React.MouseEvent) => {
    e.preventDefault();
    setIsDragging(true);
    startXRef.current = e.clientX;
  };

  useEffect(() => {
    if (!isDragging) return;

    const handleMouseMove = (e: MouseEvent) => {
      const delta = e.clientX - startXRef.current;
      startXRef.current = e.clientX;
      onResize(side === 'left' ? delta : -delta);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
    };
  }, [isDragging, onResize, side]);

  return (
    <div
      onMouseDown={handleMouseDown}
      className={cn(
        "absolute top-0 bottom-0 w-2 z-30 cursor-col-resize group flex items-center justify-center",
        side === 'left' ? 'right-0 translate-x-1/2' : 'left-0 -translate-x-1/2',
        isDragging && "bg-primary/20"
      )}
    >
      <div className={cn(
        "w-1 h-12 rounded-full bg-border transition-all duration-200",
        "group-hover:bg-primary/50 group-hover:h-20",
        isDragging && "bg-primary h-24"
      )}>
        <GripVertical className={cn(
          "w-4 h-4 text-muted-foreground/50 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-0 transition-opacity",
          "group-hover:opacity-100",
          isDragging && "opacity-100 text-primary"
        )} />
      </div>
    </div>
  );
}

function App() {
  const [sample, setSample] = useState<NextSampleResponse | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  const [serverStatus, setServerStatus] = useState<StatusResponse | null>(null);

  const [history, setHistory] = useState<{image: any, label: string}[]>([]);
  const [viewIndex, setViewIndex] = useState(0);

  const [inputValue, setInputValue] = useState("");

  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);

  const [showScatter, setShowScatter] = useState(true);
  const [showLogs, setShowLogs] = useState(true);

  // Resizable panel widths
  const [leftPanelWidth, setLeftPanelWidth] = useState(400);
  const [rightPanelWidth, setRightPanelWidth] = useState(320);

  // KNN config
  const [knnConfig, setKnnConfigState] = useState<KnnConfig | null>(null);

  const MIN_PANEL_WIDTH = 200;
  const MAX_PANEL_WIDTH = 600;

  const handleLeftResize = useCallback((delta: number) => {
    setLeftPanelWidth(prev => Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, prev + delta)));
  }, []);

  const handleRightResize = useCallback((delta: number) => {
    setRightPanelWidth(prev => Math.min(MAX_PANEL_WIDTH, Math.max(MIN_PANEL_WIDTH, prev + delta)));
  }, []);

  const handleKnnChange = useCallback(async (newK: number) => {
    try {
      const config = await setKnnConfig(newK);
      setKnnConfigState(config);
    } catch (e) {
      console.error("Failed to update KNN config:", e);
    }
  }, []);

  const inputRef = useRef<HTMLInputElement>(null);

  const isViewingHistory = viewIndex < history.length;
  const currentItem = isViewingHistory ? history[viewIndex] : sample;
  const currentImageId = currentItem?.image?.id;
  const hasSuggestion = !isViewingHistory && !!sample?.suggestion;
  const historyLabel = isViewingHistory ? history[viewIndex].label : null;

  const totalLabelled = points.filter(p => p.label).length;

  const loadPoints = async (includePredictions: boolean = true) => {
      try {
          const pts = await fetchPoints(includePredictions);
          setPoints(pts);
      } catch (e) {
          console.error("Failed to load points:", e);
      }
  };

  const loadNext = useCallback(async () => {
    try {
      const data = await fetchNextSample();
      setSample(data);
      
      if (data.status === "done") {
        return;
      }

      
      // Merge backend suggestions with remaining classes
      let newSuggestions: string[] = [];
      if (data.suggestions && data.suggestions.length > 0) {
        newSuggestions = [...data.suggestions];
        // Add remaining classes that are not in suggestions
        const remaining = CLASSES.filter(c => !newSuggestions.includes(c));
        newSuggestions = [...newSuggestions, ...remaining];
      } else {
        newSuggestions = CLASSES;
      }
      
      setInputValue("");
      setSuggestions(newSuggestions);
      setSelectedIndex(0);
      
      setTimeout(() => inputRef.current?.focus(), 50);

    } catch (e) {
      console.error(e);
    }
  }, []);

  // Poll server status until ready
  useEffect(() => {
    const pollStatus = async () => {
      try {
        const status = await fetchStatus();
        setServerStatus(status);
        if (!status.ready) {
          // Poll every 500ms while not ready
          setTimeout(pollStatus, 500);
        }
      } catch (e) {
        // Server not responding yet, retry
        setTimeout(pollStatus, 1000);
      }
    };
    pollStatus();
  }, []);

  // Load data once server is ready
  useEffect(() => {
    if (serverStatus?.ready) {
      loadNext();
      loadPoints();
      // Load KNN config
      fetchKnnConfig().then(setKnnConfigState).catch(console.error);
    }
  }, [serverStatus?.ready, loadNext]);

  useEffect(() => {
      if (isViewingHistory && historyLabel) {
          setInputValue(historyLabel);
      } else if (!isViewingHistory) {
          setInputValue("");
      }
  }, [viewIndex, history, historyLabel, isViewingHistory]);

  useEffect(() => {
    if (!inputValue) {
        // When input is empty, show the sorted suggestions from the sample (if available)
        // calculated in loadNext
        if (sample && sample.suggestions && sample.suggestions.length > 0) {
             let newSuggestions = [...sample.suggestions];
             const remaining = CLASSES.filter(c => !newSuggestions.includes(c));
             newSuggestions = [...newSuggestions, ...remaining];
             setSuggestions(newSuggestions);
        } else {
             setSuggestions(CLASSES);
        }
        setSelectedIndex(0);
        return;
    }

    const filtered = CLASSES.filter(c => 
      c.toLowerCase().includes(inputValue.toLowerCase())
    );
    // If filtering, we probably want to respect the original sort order? 
    // Or maybe we still prioritize the "likely" ones?
    // For now, let's just stick to the static list filter behavior when typing, 
    // as "suggestions" are mostly for the "empty state" shortcuts.
    setSuggestions(filtered);
    setSelectedIndex(0); 
  }, [inputValue, sample]);

  // Debounced refresh for predictions (avoid hammering backend)
  const predictionRefreshTimeout = useRef<NodeJS.Timeout | null>(null);

  const schedulePredictionRefresh = () => {
    if (predictionRefreshTimeout.current) {
      clearTimeout(predictionRefreshTimeout.current);
    }
    predictionRefreshTimeout.current = setTimeout(() => {
      loadPoints(true);
    }, 1000); // Refresh predictions 1s after last label
  };

  const handleLabel = async (label: string) => {
    if (!currentItem?.image) return;

    sendLabel({
      image_id: currentItem.image.id,
      label: label
    });

    // Immediately update local state for the labeled point
    setPoints(prev => prev.map(p => {
        if (p.id === currentItem.image.id) {
            return { ...p, label: label, predicted_label: label, confidence: 1.0 };
        }
        return p;
    }));

    // Schedule a debounced refresh to update all predictions
    schedulePredictionRefresh();

    if (isViewingHistory) {
        const newHistory = [...history];
        newHistory[viewIndex] = { ...newHistory[viewIndex], label: label };
        setHistory(newHistory);
        setViewIndex(v => v + 1);
    } else {
        const newItem = { image: sample!.image, label: label };
        setHistory(prev => [...prev, newItem]);
        setViewIndex(v => v + 1);
        await loadNext();
    }
  };

  const onInputSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const selected = suggestions[selectedIndex];
    const valueToSubmit = selected || inputValue.trim();
    if (!valueToSubmit) return;
    handleLabel(valueToSubmit);
  };

  const onInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => (prev + 1) % suggestions.length);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => (prev - 1 + suggestions.length) % suggestions.length);
    } else if (e.key === 'Tab') {
      e.preventDefault();
      if (suggestions[selectedIndex]) {
        setInputValue(suggestions[selectedIndex]);
      }
    } else if (e.key >= '1' && e.key <= '9') {
        // Quick select with number keys 1-9
        const idx = parseInt(e.key) - 1;
        if (idx < suggestions.length) {
            e.preventDefault();
            handleLabel(suggestions[idx]);
        }
    }
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (sample?.status === 'done') return;

      if (e.shiftKey && e.key === 'ArrowLeft') {
          e.preventDefault();
          setViewIndex(v => Math.max(0, v - 1));
          return;
      }
      if (e.shiftKey && e.key === 'ArrowRight') {
          e.preventDefault();
          setViewIndex(v => Math.min(history.length, v + 1));
          return;
      }

      if (e.key === 'Enter' && e.shiftKey) {
        e.preventDefault();
        
        if (isViewingHistory) {
             const selected = suggestions[selectedIndex];
             const val = selected || inputValue.trim();
             if (val) handleLabel(val);
             return;
        }

        if (hasSuggestion) {
            handleLabel(sample!.suggestion!);
            return;
        }
        
        const selected = suggestions[selectedIndex];
        const val = selected || inputValue.trim();
        if (val) {
            handleLabel(val);
        }
        return;
      }

      // 'I' key to focus input when not already focused
      if (!isViewingHistory && document.activeElement !== inputRef.current) {
          if (e.key.toLowerCase() === 'i') {
              inputRef.current?.focus();
          }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [sample, inputValue, hasSuggestion, isViewingHistory, history.length, viewIndex, suggestions, selectedIndex]); 

  // Show loading screen while server initializes
  if (!serverStatus?.ready) {
    return (
      <div className="flex h-screen w-full items-center justify-center bg-background text-foreground font-sans">
        <div className="flex flex-col items-center gap-8 max-w-md px-8">
          {/* Animated Logo/Spinner */}
          <div className="relative">
            <div className="w-24 h-24 rounded-3xl bg-gradient-to-br from-primary/20 to-primary/5 flex items-center justify-center">
              <Loader2 className="w-12 h-12 text-primary animate-spin" />
            </div>
          </div>

          {/* Status Text */}
          <div className="text-center space-y-2">
            <h1 className="text-2xl font-semibold tracking-tight">QuickSort</h1>
            <p className="text-muted-foreground text-sm">
              {serverStatus?.message || "Connecting to server..."}
            </p>
          </div>

          {/* Progress Bar */}
          <div className="w-full space-y-2">
            <div className="h-2 w-full bg-secondary rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-300 ease-out rounded-full"
                style={{ width: `${serverStatus?.progress || 0}%` }}
              />
            </div>
            <div className="flex justify-between text-xs text-muted-foreground">
              <span className="capitalize">{serverStatus?.stage || "connecting"}</span>
              <span>{serverStatus?.progress || 0}%</span>
            </div>
          </div>

          {/* Stage indicator */}
          <div className="flex gap-2">
            {["embedding", "indexing", "projection", "ready"].map((stage, idx) => (
              <div
                key={stage}
                className={cn(
                  "w-2 h-2 rounded-full transition-colors duration-300",
                  serverStatus?.stage === stage ? "bg-primary" :
                  ["embedding", "indexing", "projection", "ready"].indexOf(serverStatus?.stage || "") > idx
                    ? "bg-primary/50"
                    : "bg-muted"
                )}
              />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background text-foreground font-sans selection:bg-primary/30">

      {/* Sidebar / Scatter Plot */}
      <div
        data-testid="sidebar"
        className={cn(
          "relative flex flex-col border-r bg-background/50 backdrop-blur-xl transition-all duration-300 ease-out z-20",
          !showScatter && 'w-0 border-none overflow-hidden'
        )}
        style={{ width: showScatter ? leftPanelWidth : 0 }}
      >
        <div className={cn("absolute top-6 left-6 z-10 transition-opacity duration-300", showScatter ? 'opacity-100' : 'opacity-0')}>
            <Badge variant="outline" className="gap-2 pl-1 pr-3 py-1 bg-background/50 backdrop-blur border-border/50 shadow-sm">
                <div className="h-2 w-2 rounded-full bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.5)] animate-pulse"></div>
                <span className="text-[10px] font-semibold tracking-widest uppercase">Embedding Space</span>
            </Badge>
        </div>

        <div className="flex-1 relative overflow-hidden">
           {showScatter && <ScatterPlot points={points} currentImageId={currentImageId} />}
        </div>

        <div className={cn("p-6 border-t flex flex-col gap-4 transition-opacity duration-300", showScatter ? 'opacity-100' : 'opacity-0')}>
            <div className="flex justify-between items-center text-xs text-muted-foreground">
                <span>Projection</span>
                <span className="font-mono text-foreground">UMAP</span>
            </div>
            <div className="flex justify-between items-center text-xs text-muted-foreground">
                <span>Points</span>
                <span className="font-mono text-foreground">{points.length}</span>
            </div>
        </div>

        {/* Resize Handle for Left Panel */}
        {showScatter && <ResizeHandle onResize={handleLeftResize} side="left" />}
      </div>

      {/* Main Content */}
      <div className="flex-1 h-full flex flex-col relative bg-background">
        
        {/* Header */}
        <div className="w-full h-16 px-6 flex justify-between items-center border-b bg-background/80 backdrop-blur-sm z-30">
          <div className="flex items-center gap-4">
             <Button
                variant="ghost" 
                size="icon"
                title="Toggle Scatter Plot"
                onClick={() => setShowScatter(!showScatter)}
                className={cn("text-muted-foreground hover:text-foreground", showScatter && "bg-accent text-accent-foreground")}
             >
                <Layout size={18} />
             </Button>
             
             <div className="h-4 w-px bg-border"></div>

             <div>
                <h1 className="text-sm font-semibold tracking-wide flex items-center gap-2">
                    QuickSort
                    <span className="text-muted-foreground text-xs font-normal">/</span>
                    <span className="text-muted-foreground font-normal">Labelling</span>
                </h1>
             </div>
          </div>

          <div className="flex items-center gap-6">
             <div className="flex flex-col items-end">
                <div className="flex items-baseline gap-1">
                    <span className="text-xl font-light tracking-tight">{totalLabelled}</span>
                    <span className="text-xs text-muted-foreground font-light">/ {points.length}</span>
                </div>
                <span className="text-[10px] text-muted-foreground uppercase tracking-widest font-medium">Labelled</span>
             </div>

             <div className="h-4 w-px bg-border"></div>

             {/* KNN K Slider */}
             {knnConfig && (
               <div className="flex items-center gap-3">
                 <Settings2 size={14} className="text-muted-foreground" />
                 <div className="flex flex-col gap-1">
                   <div className="flex items-center gap-2">
                     <span className="text-[10px] text-muted-foreground uppercase tracking-widest font-medium">K</span>
                     <input
                       type="range"
                       min={knnConfig.min_k}
                       max={knnConfig.max_k}
                       value={knnConfig.k_neighbors}
                       onChange={(e) => handleKnnChange(parseInt(e.target.value))}
                       className="w-20 h-1 bg-secondary rounded-full appearance-none cursor-pointer accent-primary"
                       title={`K neighbors: ${knnConfig.k_neighbors}`}
                     />
                     <span className="text-xs font-mono text-foreground w-6">{knnConfig.k_neighbors}</span>
                   </div>
                 </div>
               </div>
             )}

             <div className="h-4 w-px bg-border"></div>

             <Button
                variant="ghost"
                size="icon"
                title="Toggle Logs"
                onClick={() => setShowLogs(!showLogs)}
                className={cn("text-muted-foreground hover:text-foreground", showLogs && "bg-accent text-accent-foreground")}
             >
                <Terminal size={18} />
             </Button>
          </div>
        </div>

        {/* Central Stage */}
        <div className="flex-1 relative flex flex-col items-center justify-center p-8 overflow-hidden">
            
            {/* Background Glow */}
            <div className={cn(
                "absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-primary/5 blur-[120px] rounded-full pointer-events-none transition-all duration-1000",
                hasSuggestion ? 'bg-purple-500/5' : ''
            )}></div>

            {/* Navigation Buttons (Floating) */}
            <Button
                variant="secondary"
                size="icon"
                onClick={() => setViewIndex(v => Math.max(0, v - 1))}
                disabled={viewIndex === 0}
                className="absolute left-12 top-1/2 -translate-y-1/2 h-12 w-12 rounded-full shadow-lg opacity-0 hover:opacity-100 disabled:opacity-0 transition-opacity duration-300"
            >
                <ArrowLeft size={20} />
            </Button>
            <Button
                variant="secondary"
                size="icon"
                onClick={() => setViewIndex(v => Math.min(history.length, v + 1))}
                disabled={viewIndex === history.length}
                className="absolute right-12 top-1/2 -translate-y-1/2 h-12 w-12 rounded-full shadow-lg opacity-0 hover:opacity-100 disabled:opacity-0 transition-opacity duration-300"
            >
                <ChevronRight size={20} />
            </Button>

            {sample?.status === "done" && !isViewingHistory ? (
                <div className="text-center p-16 animate-in zoom-in duration-500 bg-card rounded-3xl border shadow-2xl">
                    <div className="w-20 h-20 bg-primary/10 rounded-2xl flex items-center justify-center mb-8 text-primary mx-auto">
                        <Check size={40} />
                    </div>
                    <h1 className="text-4xl font-bold mb-4 tracking-tight">All Done</h1>
                    <p className="text-muted-foreground text-lg">The entire dataset has been labeled.</p>
                </div>
            ) : (
                <div className="flex flex-col items-center w-full max-w-2xl relative z-10">
                    
                    {/* Status Pill */}
                    <div className="mb-8">
                        {isViewingHistory ? (
                            <Badge variant="outline" className="gap-2 py-1.5 px-4 text-orange-400 border-orange-500/20 bg-orange-500/10">
                                <div className="w-1.5 h-1.5 rounded-full bg-orange-500 animate-pulse"></div>
                                REVIEWING HISTORY ({viewIndex + 1} / {history.length + 1})
                            </Badge>
                        ) : hasSuggestion ? (
                            <Badge variant="outline" className="gap-2 py-1.5 px-4 text-purple-400 border-purple-500/20 bg-purple-500/10">
                                <Sparkles size={12} className="fill-purple-400" />
                                AI SUGGESTION AVAILABLE
                            </Badge>
                        ) : (
                            <Badge variant="outline" className="gap-2 py-1.5 px-4 text-blue-400 border-blue-500/20 bg-blue-500/10">
                                <div className="w-1.5 h-1.5 rounded-full bg-blue-500"></div>
                                AWAITING INPUT
                            </Badge>
                        )}
                    </div>

                    {/* Image Container */}
                    <div className="relative group mb-12">
                        {/* Dynamic Shadow */}
                        <div className="absolute -inset-0.5 bg-gradient-to-b from-primary/20 to-transparent rounded-[2rem] opacity-50 blur-sm group-hover:opacity-75 transition duration-500"></div>
                        
                        <Card className="relative w-[340px] h-[340px] border-border/50 bg-card/50 backdrop-blur-sm shadow-2xl flex items-center justify-center overflow-hidden rounded-[1.8rem]">
                            {currentItem?.image ? (
                                <img 
                                src={`data:image/png;base64,${currentItem.image.data}`} 
                                alt="To label" 
                                className="w-full h-full object-contain p-8 transition-transform duration-700 group-hover:scale-105"
                                style={{ imageRendering: 'pixelated' }} 
                                />
                            ) : (
                                <div className="w-12 h-12 border-2 border-border border-t-primary rounded-full animate-spin"></div>
                            )}
                            
                            {/* ID Badge */}
                            <div className="absolute bottom-4 right-4">
                                <Badge variant="secondary" className="text-[10px] font-mono opacity-0 group-hover:opacity-100 transition-opacity">
                                    {currentItem?.image?.id.slice(0,8)}
                                </Badge>
                            </div>
                        </Card>
                    </div>

                    {/* Interaction Area */}
                    <div className="w-full relative">
                        
                        {/* Suggestion Prompt */}
                        {hasSuggestion && !isViewingHistory && (
                            <div className="absolute -top-20 left-0 right-0 flex justify-center pointer-events-none">
                                <Card className="px-6 py-3 rounded-2xl shadow-xl flex items-center gap-4 animate-in slide-in-from-bottom-4 fade-in duration-300 border-primary/20 bg-card/80 backdrop-blur-xl">
                                    <span className="text-muted-foreground text-xs font-medium uppercase tracking-wider">Is it</span>
                                    <span className="text-2xl font-bold font-mono tracking-tight">{sample?.suggestion}</span>
                                    <span className="text-muted-foreground text-xl font-light">?</span>
                                    <div className="h-6 w-px bg-border mx-2"></div>
                                    <div className="flex items-center gap-2">
                                        <kbd className="hidden sm:inline-flex items-center h-6 px-2 text-[10px] font-medium text-muted-foreground bg-muted border rounded-md shadow-sm font-sans">Shift</kbd>
                                        <kbd className="hidden sm:inline-flex items-center h-6 px-2 text-[10px] font-medium text-muted-foreground bg-muted border rounded-md shadow-sm font-sans">Enter</kbd>
                                        <span className="text-[10px] text-muted-foreground font-medium uppercase">Accept</span>
                                    </div>
                                </Card>
                            </div>
                        )}

                        {/* Input Field */}
                        <form onSubmit={onInputSubmit} className="relative w-full max-w-md mx-auto group">
                            <div className="absolute inset-y-0 left-4 flex items-center pointer-events-none z-20">
                                <Search size={18} className={cn("transition-colors duration-300", inputValue ? 'text-foreground' : 'text-muted-foreground')} />
                            </div>
                            
                            <Input
                                ref={inputRef}
                                type="text"
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onKeyDown={onInputKeyDown}
                                placeholder={isViewingHistory ? "Type correction..." : hasSuggestion ? "Or type to correct..." : "Type label..."}
                                className="pl-12 pr-12 h-14 text-lg rounded-2xl bg-secondary/50 backdrop-blur border-border/50 focus-visible:ring-offset-0 focus-visible:ring-1 focus-visible:ring-primary/50 focus-visible:border-primary/50 shadow-lg"
                                autoFocus
                            />

                            {/* Right Action Icon */}
                            <div className="absolute inset-y-0 right-3 flex items-center z-20">
                                <Button 
                                    type="submit"
                                    size="icon"
                                    variant="ghost"
                                    className={cn("h-8 w-8", inputValue && "text-primary hover:text-primary hover:bg-primary/10")}
                                    disabled={!inputValue.trim() && suggestions.length === 0}
                                >
                                    <ChevronRight size={18} />
                                </Button>
                            </div>

                            {/* Dropdown */}
                            {suggestions.length > 0 && (
                                <div className="absolute top-full left-0 right-0 mt-2 bg-popover/95 backdrop-blur-xl border rounded-xl overflow-hidden shadow-2xl z-50 max-h-60 overflow-y-auto animate-in fade-in slide-in-from-top-1 duration-200 p-1">
                                    <div className="px-3 py-2 text-[10px] font-semibold text-muted-foreground uppercase tracking-wider sticky top-0 bg-popover/95 backdrop-blur">Suggestions</div>
                                    {suggestions.map((suggestion, idx) => (
                                        <div 
                                            key={suggestion}
                                            className={cn(
                                                "px-3 py-2.5 mx-1 rounded-lg cursor-pointer text-sm font-medium transition-all flex justify-between items-center",
                                                idx === selectedIndex ? 'bg-accent text-accent-foreground' : 'text-muted-foreground hover:bg-accent/50 hover:text-accent-foreground'
                                            )}
                                            onClick={() => {
                                                setInputValue(suggestion);
                                                handleLabel(suggestion);
                                            }}
                                            onMouseEnter={() => setSelectedIndex(idx)}
                                        >
                                            <div className="flex items-center gap-2">
                                                {idx < 9 && (
                                                    <span className="font-mono text-xs text-muted-foreground/50 w-4 text-center">
                                                        {idx + 1}
                                                    </span>
                                                )}
                                                <span className="font-mono tracking-wide">{suggestion}</span>
                                            </div>
                                            {idx === selectedIndex && (
                                                <div className="flex items-center gap-1.5">
                                                    <span className="text-[9px] text-muted-foreground uppercase tracking-wider font-semibold">Select</span>
                                                    <ChevronRight size={12} className="text-muted-foreground" />
                                                </div>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            )}
                        </form>

                        <div className="mt-8 flex justify-center gap-8 opacity-40 hover:opacity-100 transition-opacity duration-500">
                            <div className="flex items-center gap-2 text-[10px] text-muted-foreground font-medium">
                                <div className="flex gap-1">
                                    <kbd className="bg-muted px-1.5 py-0.5 rounded border font-sans">Shift</kbd>
                                    <kbd className="bg-muted px-1.5 py-0.5 rounded border font-sans">←</kbd>
                                </div>
                                <span>History</span>
                            </div>
                            <div className="flex items-center gap-2 text-[10px] text-muted-foreground font-medium">
                                <div className="flex gap-1">
                                    <kbd className="bg-muted px-1.5 py-0.5 rounded border font-sans">Shift</kbd>
                                    <kbd className="bg-muted px-1.5 py-0.5 rounded border font-sans">Enter</kbd>
                                </div>
                                <span>Accept</span>
                            </div>
                        </div>

                    </div>
                </div>
            )}
        </div>
      </div>

      {/* Logs Panel */}
      {showLogs && (
         <div
           className="bg-card border-l relative flex-shrink-0 transition-all duration-300 z-20 flex flex-col"
           style={{ width: rightPanelWidth }}
         >
             {/* Resize Handle for Right Panel */}
             <ResizeHandle onResize={handleRightResize} side="right" />

             <div className="h-12 border-b flex items-center px-4 bg-card/50 backdrop-blur">
                 <span className="text-xs font-semibold text-muted-foreground uppercase tracking-widest">System Logs</span>
             </div>
             <div className="flex-1 overflow-hidden">
                <LogConsole />
             </div>
         </div>
      )}

    </div>
  );
}

export default App;
