import { useEffect, useState, useRef, useCallback } from 'react';
import './App.css';
import { LogConsole } from './components/LogConsole';
import { ScatterPlot } from './components/ScatterPlot';
import { fetchNextSample, sendLabel, fetchPoints, type NextSampleResponse, type Point } from './api';
import { Check, ChevronRight, Layout, Terminal, Maximize2, Zap, ArrowLeft } from 'lucide-react';

const CLASSES = [
  "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"
];

function App() {
  const [sample, setSample] = useState<NextSampleResponse | null>(null);
  const [points, setPoints] = useState<Point[]>([]);
  
  // History State
  const [history, setHistory] = useState<{image: any, label: string}[]>([]);
  const [viewIndex, setViewIndex] = useState(0); // 0 to history.length (where length means "new")

  // Interaction State
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);
  
  // Autocomplete State
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  
  // Layout State
  const [showScatter, setShowScatter] = useState(true);
  const [showLogs, setShowLogs] = useState(true);
  
  const inputRef = useRef<HTMLInputElement>(null);

  // Derived State
  const isViewingHistory = viewIndex < history.length;
  // If we are viewing history, use that item. If we are at the "end", use the new sample.
  const currentItem = isViewingHistory ? history[viewIndex] : sample;
  const currentImageId = currentItem?.image?.id;
  
  // Suggestion logic only applies for new items (sample)
  // For history items, we just show the label they already have (and allow edit)
  const hasSuggestion = !isViewingHistory && !!sample?.suggestion;
  
  // If viewing history, the "suggestion" is actually the label we gave it.
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

  // Initial load
  useEffect(() => {
    loadNext();
    loadPoints();
  }, [loadNext]);

  // Sync viewIndex with history length when new history added?
  // No, viewIndex should start at 0 (if history empty) which matches history.length.
  
  // When switching views (history vs new), update input value
  useEffect(() => {
      if (isViewingHistory && historyLabel) {
          setInputValue(historyLabel);
      } else if (!isViewingHistory) {
          setInputValue(""); // Clear for new input
      }
  }, [viewIndex, history, historyLabel, isViewingHistory]);

  // Autocomplete logic
  useEffect(() => {
    const filtered = CLASSES.filter(c => 
      c.toLowerCase().includes(inputValue.toLowerCase())
    );
    setSuggestions(filtered);
    setSelectedIndex(0); 
  }, [inputValue]);

  const handleLabel = async (label: string) => {
    if (!currentItem?.image) return;

    // Send to backend
    sendLabel({
      image_id: currentItem.image.id,
      label: label
    });
    
    // Update local points immediately
    setPoints(prev => prev.map(p => {
        if (p.id === currentItem.image.id) {
            return { ...p, label: label };
        }
        return p;
    }));

    if (isViewingHistory) {
        // We are editing history.
        // Update the history item
        const newHistory = [...history];
        newHistory[viewIndex] = { ...newHistory[viewIndex], label: label };
        setHistory(newHistory);
        
        // Advance to next item (either next history or new)
        setViewIndex(v => v + 1);
    } else {
        // We are labelling a NEW item.
        // Add to history
        const newItem = { image: sample!.image, label: label };
        setHistory(prev => [...prev, newItem]);
        
        // Advance view index (to match new length)
        setViewIndex(v => v + 1);
        
        // Fetch next
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

  // Navigation & Shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (sample?.status === 'done') return;

      // Navigation: Left/Right Arrows
      // Only navigate if input is empty OR modifiers used?
      // Standard behavior: Left/Right in input moves cursor.
      // We should use Modifier+Arrow OR verify input state.
      // Let's use Alt+Arrow or just Arrow if not focused?
      // User said "toggle back with the arrows".
      // If the input is focused, Left/Right is needed for text.
      // Let's try: If text cursor is at start/end? Or just use Alt+Left/Right.
      // Or: If input is empty?
      // Let's implement: Alt + Left / Alt + Right for safe navigation.
      
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

      // Shift+Enter (Accept Suggestion / Submit)
      if (e.key === 'Enter' && e.shiftKey) {
        e.preventDefault();
        
        // If viewing history, accept current input as "correction"
        if (isViewingHistory) {
             const selected = suggestions[selectedIndex];
             const val = selected || inputValue.trim();
             if (val) handleLabel(val);
             return;
        }

        // If new item and suggestion exists
        if (hasSuggestion) {
            handleLabel(sample!.suggestion!);
            return;
        }
        
        // If input has value
        const selected = suggestions[selectedIndex];
        const val = selected || inputValue.trim();
        if (val) {
            handleLabel(val);
        }
        return;
      }

      // J/I Shortcuts (Only for new items with suggestion)
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
    <div className="flex h-screen w-full overflow-hidden bg-[#0f0f11] text-zinc-200 font-sans selection:bg-indigo-500/30">
      
      {showScatter && (
        <div className="w-1/3 h-full border-r border-white/5 flex flex-col relative bg-[#0a0a0c] transition-all duration-300 shadow-2xl z-10">
            <div className="absolute top-4 left-4 z-10">
                <div className="bg-black/40 backdrop-blur-md border border-white/10 px-3 py-1.5 rounded-full text-[10px] uppercase tracking-widest text-zinc-400 font-bold shadow-lg flex items-center gap-2">
                    <span className="w-1.5 h-1.5 rounded-full bg-indigo-500 animate-pulse"></span>
                    Embedding Space
                </div>
            </div>
            <div className="flex-1 relative">
               <ScatterPlot points={points} currentImageId={currentImageId} />
            </div>
        </div>
      )}

      <div className="flex-1 h-full flex flex-col relative bg-gradient-to-br from-[#121214] to-[#0c0c0e] transition-all duration-300">
        <div className="w-full h-16 px-6 flex justify-between items-center border-b border-white/5 bg-[#121214]/80 backdrop-blur-sm z-20 sticky top-0">
          <div className="flex items-center gap-5">
             <button 
                onClick={() => setShowScatter(!showScatter)}
                className={`p-2 rounded-lg transition-all duration-200 hover:scale-105 active:scale-95 ${showScatter ? 'bg-white/10 text-white shadow-lg shadow-black/20' : 'text-zinc-500 hover:bg-white/5 hover:text-zinc-300'}`}
                title="Toggle Scatter Plot"
             >
                <Layout size={18} strokeWidth={2} />
             </button>
             
             <div className="h-5 w-px bg-white/10"></div>

             <div>
                <h1 className="text-sm font-semibold text-white tracking-wide leading-none flex items-center gap-2">
                    Labelling Station
                    <span className="bg-white/5 text-zinc-500 px-1.5 rounded text-[9px] font-mono border border-white/5">v1.0</span>
                </h1>
                
                {/* Mode Indicator */}
                <div className="flex items-center gap-2 mt-1.5">
                   {isViewingHistory ? (
                       <span className="text-[10px] font-bold uppercase tracking-wider text-orange-400 flex items-center gap-1.5">
                         <span className="w-1 h-1 rounded-full bg-orange-400 shadow-[0_0_8px_rgba(251,146,60,0.5)]"></span>
                         Review Mode ({viewIndex + 1}/{history.length + 1})
                       </span>
                   ) : (
                       <span className="text-[10px] font-bold uppercase tracking-wider text-blue-400 flex items-center gap-1.5">
                         <span className="w-1 h-1 rounded-full bg-blue-400 shadow-[0_0_8px_rgba(96,165,250,0.5)]"></span>
                         Labelling Mode
                       </span>
                   )}
                </div>
             </div>
          </div>

          <div className="flex items-center gap-5">
             <div className="flex flex-col items-end">
                <span className="text-xl font-medium text-white leading-none font-mono tracking-tight">{totalLabelled}</span>
                <span className="text-[9px] text-zinc-500 uppercase tracking-widest font-semibold mt-0.5">Labelled</span>
             </div>
             
             <div className="h-5 w-px bg-white/10"></div>
             
             <button 
                onClick={() => setShowLogs(!showLogs)}
                className={`p-2 rounded-lg transition-all duration-200 hover:scale-105 active:scale-95 ${showLogs ? 'bg-white/10 text-white shadow-lg shadow-black/20' : 'text-zinc-500 hover:bg-white/5 hover:text-zinc-300'}`}
                title="Toggle Logs"
             >
                <Terminal size={18} strokeWidth={2} />
             </button>
          </div>
        </div>

        <div className="flex-1 relative flex flex-col items-center justify-center overflow-hidden">
            {/* Navigation Arrows */}
            <button 
                onClick={() => setViewIndex(v => Math.max(0, v - 1))}
                disabled={viewIndex === 0}
                className="absolute left-8 top-1/2 -translate-y-1/2 p-4 rounded-full bg-white/5 hover:bg-white/10 disabled:opacity-0 transition-all z-30"
            >
                <ArrowLeft size={24} />
            </button>
            <button 
                onClick={() => setViewIndex(v => Math.min(history.length, v + 1))}
                disabled={viewIndex === history.length}
                className="absolute right-8 top-1/2 -translate-y-1/2 p-4 rounded-full bg-white/5 hover:bg-white/10 disabled:opacity-0 transition-all z-30"
            >
                <ChevronRight size={24} />
            </button>

            {sample?.status === "done" && !isViewingHistory ? (
                <div className="text-center p-12 animate-in zoom-in duration-500 bg-white/5 rounded-3xl border border-white/5 backdrop-blur-xl shadow-2xl">
                    <div className="w-20 h-20 bg-green-500/20 rounded-2xl flex items-center justify-center mb-6 text-green-400 mx-auto shadow-[0_0_30px_rgba(74,222,128,0.1)] border border-green-500/20">
                        <Check size={40} strokeWidth={3} />
                    </div>
                    <h1 className="text-4xl font-bold text-white mb-3 tracking-tight">Mission Complete</h1>
                    <p className="text-zinc-400 text-lg font-light">All datasets have been successfully classified.</p>
                </div>
            ) : (
                <div className="flex flex-col items-center justify-center w-full max-w-3xl gap-10 p-8">
                    
                    <div className="relative group perspective-1000">
                        <div className={`absolute -inset-10 rounded-[3rem] blur-3xl opacity-10 group-hover:opacity-20 transition duration-1000 ${hasSuggestion ? 'bg-purple-500' : isViewingHistory ? 'bg-orange-500' : 'bg-blue-500'}`}></div>
                        <div className="relative rounded-2xl overflow-hidden bg-[#050505] border border-white/10 shadow-2xl shadow-black/80 group-hover:scale-[1.02] transition-transform duration-500 ease-out">
                            {currentItem?.image ? (
                                <img 
                                src={`data:image/png;base64,${currentItem.image.data}`} 
                                alt="To label" 
                                className="w-80 h-80 object-contain p-4"
                                style={{ imageRendering: 'pixelated' }} 
                                />
                            ) : (
                                <div className="w-80 h-80 flex items-center justify-center bg-[#050505]">
                                    <div className="w-12 h-12 border-2 border-white/10 border-t-indigo-500 rounded-full animate-spin"></div>
                                </div>
                            )}
                            
                            {/* Metadata Overlay */}
                            <div className="absolute top-3 right-3 bg-black/60 backdrop-blur px-2 py-1 rounded-md border border-white/5">
                                <span className="text-[10px] font-mono text-zinc-500 uppercase">ID: {currentItem?.image?.id.slice(0,4)}</span>
                            </div>
                        </div>
                    </div>

                    <div className="w-full max-w-[440px] flex flex-col items-center justify-center relative">
                        
                        {/* Suggestion Badge (Blended) - Only for New Items */}
                        {hasSuggestion && (
                            <div className="mb-6 flex flex-col items-center animate-in slide-in-from-bottom-2 fade-in">
                                <div className="text-zinc-500 text-[10px] uppercase tracking-[0.2em] font-bold mb-2">Confidence Check</div>
                                <div className="flex items-center gap-3 bg-[#1a1a1c] border border-white/10 px-5 py-2 rounded-xl shadow-lg">
                                    <span className="text-2xl font-bold text-white font-mono">{sample?.suggestion}</span>
                                    <div className="h-6 w-px bg-white/10"></div>
                                    <div className="flex gap-2">
                                        <div className="flex flex-col items-center">
                                            <span className="text-[9px] text-zinc-500 uppercase font-bold">Accept</span>
                                            <kbd className="text-[10px] text-emerald-400 font-mono bg-emerald-500/10 px-1.5 rounded border border-emerald-500/20">J</kbd>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Review Mode Banner */}
                        {isViewingHistory && (
                             <div className="mb-6 flex flex-col items-center animate-in slide-in-from-bottom-2 fade-in">
                                <div className="text-orange-400/80 text-[10px] uppercase tracking-[0.2em] font-bold mb-2">Reviewing Past Label</div>
                             </div>
                        )}

                        <form onSubmit={onInputSubmit} className="relative w-full group">
                            <div className={`absolute inset-0 bg-gradient-to-r rounded-2xl blur-lg opacity-0 group-focus-within:opacity-100 transition duration-500 ${hasSuggestion ? 'from-purple-500/20 to-pink-500/20' : isViewingHistory ? 'from-orange-500/20 to-red-500/20' : 'from-blue-500/20 to-cyan-500/20'}`}></div>
                            <input
                                ref={inputRef}
                                type="text"
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onKeyDown={onInputKeyDown}
                                placeholder={isViewingHistory ? "Correct label..." : hasSuggestion ? "Or type correction..." : "Type label..."}
                                className="w-full bg-[#151515]/80 backdrop-blur-xl text-white text-xl font-mono px-6 py-5 rounded-2xl border border-white/10 focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/20 focus:outline-none text-center placeholder-zinc-600 transition-all shadow-2xl relative z-10"
                                autoFocus
                            />
                            
                            {/* Input Icon / Button */}
                            <button 
                                type="submit"
                                className="absolute right-3 top-1/2 -translate-y-1/2 p-2 bg-white/5 text-white rounded-lg hover:bg-white/10 transition-all z-20"
                                disabled={!inputValue.trim() && suggestions.length === 0}
                            >
                                <ChevronRight size={18} strokeWidth={2} />
                            </button>

                            {/* Autocomplete Dropdown */}
                            {suggestions.length > 0 && inputValue.length > 0 && (
                                <div className="absolute top-full left-0 right-0 mt-3 bg-[#1a1a1c]/90 backdrop-blur-xl border border-white/10 rounded-xl overflow-hidden shadow-2xl z-30 max-h-56 overflow-y-auto animate-in slide-in-from-top-2">
                                <div className="sticky top-0 bg-[#1a1a1c] border-b border-white/5 px-3 py-1.5 text-[9px] uppercase tracking-wider font-bold text-zinc-500">Suggestions</div>
                                {suggestions.map((suggestion, idx) => (
                                    <div 
                                    key={suggestion}
                                    className={`px-4 py-3 cursor-pointer text-sm font-mono tracking-wide transition-colors flex justify-between items-center ${idx === selectedIndex ? 'bg-blue-600/20 text-blue-200 border-l-2 border-blue-500' : 'text-zinc-400 hover:bg-white/5 border-l-2 border-transparent'}`}
                                    onClick={() => {
                                        setInputValue(suggestion);
                                        handleLabel(suggestion);
                                    }}
                                    onMouseEnter={() => setSelectedIndex(idx)}
                                    >
                                    <span>{suggestion}</span>
                                    {idx === selectedIndex && <span className="text-[9px] text-blue-400/50 uppercase tracking-wider font-sans font-bold">Shift+Enter</span>}
                                    </div>
                                ))}
                                </div>
                            )}
                        </form>
                        
                        <div className="mt-4 text-center">
                            <span className="text-[10px] text-zinc-600 font-mono">
                                Navigate: <kbd className="bg-white/5 px-1 rounded border border-white/10">Alt</kbd> + <kbd className="bg-white/5 px-1 rounded border border-white/10">←/→</kbd>
                            </span>
                        </div>
                    </div>
                </div>
            )}
        </div>
      </div>

      {showLogs && (
         <div className="w-80 h-full bg-[#0a0a0c] border-l border-white/5 relative flex-shrink-0 transition-all duration-300 shadow-2xl z-20">
             <LogConsole />
         </div>
      )}

    </div>
  );
}

export default App;
