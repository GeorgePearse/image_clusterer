import { useEffect, useState, useRef, useCallback } from 'react';
import './App.css';
import { LogConsole } from './components/LogConsole';
import { ScatterPlot } from './components/ScatterPlot';
import { fetchNextSample, sendLabel, fetchPoints, type NextSampleResponse, type Point } from './api';
import { Check, ChevronRight, Layout, Terminal, Sparkles, ArrowLeft, Search } from 'lucide-react';
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

const CLASSES = [
  "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
];

function App() {
  const [sample, setSample] = useState<NextSampleResponse | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  
  const [history, setHistory] = useState<{image: any, label: string}[]>([]);
  const [viewIndex, setViewIndex] = useState(0);

  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  
  const [showScatter, setShowScatter] = useState(true);
  const [showLogs, setShowLogs] = useState(true);
  
  const inputRef = useRef<HTMLInputElement>(null);

  const isViewingHistory = viewIndex < history.length;
  const currentItem = isViewingHistory ? history[viewIndex] : sample;
  const currentImageId = currentItem?.image?.id;
  const hasSuggestion = !isViewingHistory && !!sample?.suggestion;
  const historyLabel = isViewingHistory ? history[viewIndex].label : null;

  const totalLabelled = points.filter(p => p.label).length;

  const loadPoints = async () => {
      try {
          const pts = await fetchPoints();
          setPoints(pts);
      } catch (e) {
          console.error("Failed to load points:", e);
      }
  };

  const loadNext = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchNextSample();
      setSample(data);
      
      if (data.status === "done") {
        return;
      }

      setInputValue("");
      setSuggestions(CLASSES);
      setSelectedIndex(0);
      
      setTimeout(() => inputRef.current?.focus(), 50);

    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadNext();
    loadPoints();
  }, [loadNext]);

  useEffect(() => {
      if (isViewingHistory && historyLabel) {
          setInputValue(historyLabel);
      } else if (!isViewingHistory) {
          setInputValue("");
      }
  }, [viewIndex, history, historyLabel, isViewingHistory]);

  useEffect(() => {
    const filtered = CLASSES.filter(c => 
      c.toLowerCase().includes(inputValue.toLowerCase())
    );
    setSuggestions(filtered);
    setSelectedIndex(0); 
  }, [inputValue]);

  const handleLabel = async (label: string) => {
    if (!currentItem?.image) return;

    sendLabel({
      image_id: currentItem.image.id,
      label: label
    });
    
    setPoints(prev => prev.map(p => {
        if (p.id === currentItem.image.id) {
            return { ...p, label: label };
        }
        return p;
    }));

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
    }
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (sample?.status === 'done') return;

      if (e.altKey && e.key === 'ArrowLeft') {
          e.preventDefault();
          setViewIndex(v => Math.max(0, v - 1));
          return;
      }
      if (e.altKey && e.key === 'ArrowRight') {
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

      if (!isViewingHistory && hasSuggestion && document.activeElement !== inputRef.current) {
          if (e.key.toLowerCase() === 'j') {
              e.preventDefault(); 
              handleLabel(sample!.suggestion!);
          } else if (e.key.toLowerCase() === 'i') {
              inputRef.current?.focus();
          }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [sample, inputValue, hasSuggestion, isViewingHistory, history.length, viewIndex, suggestions, selectedIndex]); 

  return (
    <div className="flex h-screen w-full overflow-hidden bg-background text-foreground font-sans selection:bg-primary/30">
      
      {/* Sidebar / Scatter Plot */}
      <div 
        className={cn(
          "relative flex flex-col border-r bg-background/50 backdrop-blur-xl transition-all duration-500 ease-out z-20",
          showScatter ? 'w-[400px]' : 'w-0 border-none'
        )}
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
      </div>

      {/* Main Content */}
      <div className="flex-1 h-full flex flex-col relative bg-background">
        
        {/* Header */}
        <div className="w-full h-16 px-6 flex justify-between items-center border-b bg-background/80 backdrop-blur-sm z-30">
          <div className="flex items-center gap-4">
             <Button
                variant="ghost" 
                size="icon"
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
             
             <Button 
                variant="ghost"
                size="icon"
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
                                        <kbd className="hidden sm:inline-flex items-center h-6 px-2 text-[10px] font-medium text-muted-foreground bg-muted border rounded-md shadow-sm font-sans">J</kbd>
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
                            {suggestions.length > 0 && inputValue.length > 0 && (
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
                                            <span className="font-mono tracking-wide">{suggestion}</span>
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
                                    <kbd className="bg-muted px-1.5 py-0.5 rounded border font-sans">Alt</kbd>
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
         <div className="w-[320px] bg-card border-l relative flex-shrink-0 transition-all duration-300 z-20 flex flex-col">
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
