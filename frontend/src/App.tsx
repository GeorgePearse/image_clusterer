import { useEffect, useState, useCallback } from 'react';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import './App.css';
import { LogConsole } from './components/LogConsole';
import { type ImagePair, fetchPair, sendVote } from './api';

interface HistoryItem {
  pair: ImagePair;
  vote: boolean | null;
}

function App() {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [currentIndex, setCurrentIndex] = useState(-1);
  const [loading, setLoading] = useState(false);

  const loadNewPair = useCallback(async () => {
    if (loading) return;
    setLoading(true);
    try {
      const pair = await fetchPair();
      setHistory(prev => {
        const newHistory = [...prev, { pair, vote: null }];
        // Ensure we update current index to point to the new item if we were at the end
        // But we handle this in handleVote/goForward usually.
        // If it's the first item:
        if (prev.length === 0) return newHistory;
        return newHistory;
      });
      // If we are initializing
      if (currentIndex === -1) setCurrentIndex(0);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  }, [loading, currentIndex]);

  // Initial load
  useEffect(() => {
    // Only load if empty
    if (history.length === 0) {
      loadNewPair();
    }
  }, []); 

  const handleVote = useCallback(async (isSame: boolean) => {
    if (currentIndex === -1 || loading) return;

    const currentItem = history[currentIndex];
    const previousVote = currentItem.vote;

    // Optimistic update
    setHistory(prev => {
      const newHistory = [...prev];
      newHistory[currentIndex] = { ...newHistory[currentIndex], vote: isSame };
      return newHistory;
    });

    if (previousVote !== isSame) {
      // Send vote in background
      sendVote({
        id1: currentItem.pair.image1.id,
        id2: currentItem.pair.image2.id,
        are_same: isSame
      }).catch(console.error);
    }

    // Advance
    if (currentIndex === history.length - 1) {
      // We need to fetch the next one
      setLoading(true); // Set loading explicitly before async call
      try {
        const pair = await fetchPair();
        setHistory(prev => [...prev, { pair, vote: null }]);
        setCurrentIndex(prev => prev + 1);
      } finally {
        setLoading(false);
      }
    } else {
      setCurrentIndex(prev => prev + 1);
    }
  }, [currentIndex, history, loading]);

  const goBack = useCallback(() => {
    if (currentIndex > 0) setCurrentIndex(prev => prev - 1);
  }, [currentIndex]);

  const goForward = useCallback(async () => {
    if (currentIndex < history.length - 1) {
      setCurrentIndex(prev => prev + 1);
    } else if (currentIndex === history.length - 1 && history[currentIndex].vote !== null) {
      // Fetch new if we are at end and voted
      setLoading(true);
      try {
        const pair = await fetchPair();
        setHistory(prev => [...prev, { pair, vote: null }]);
        setCurrentIndex(prev => prev + 1);
      } finally {
        setLoading(false);
      }
    }
  }, [currentIndex, history]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Ignore if modifiers used
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      
      switch(e.key.toLowerCase()) {
        case 'j':
          handleVote(true);
          break;
        case 'i':
          handleVote(false);
          break;
        case 'arrowleft':
          goBack();
          break;
        case 'arrowright':
          goForward();
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleVote, goBack, goForward]);

  // Helper to get current item safely
  const currentItem = history[currentIndex];

  if (!currentItem) return (
    <div className="flex h-screen w-full bg-[#242424]">
      <LogConsole />
      <div className="flex-1 flex items-center justify-center">
        <div style={{ opacity: 0.5 }}>Loading...</div>
      </div>
    </div>
  );

  return (
    <div className="flex h-screen w-full overflow-hidden bg-[#242424]">
      <LogConsole />
      
      <div className="flex-1 relative overflow-hidden">
        <div className="container h-full">
          <h1>Are they the same class?</h1>

          <button className="nav-btn nav-prev" onClick={goBack} disabled={currentIndex === 0}>
            <ChevronLeft size={32} />
          </button>
          
          <button className="nav-btn nav-next" onClick={goForward}>
            <ChevronRight size={32} />
          </button>

          <div className={`image-pair ${loading ? 'opacity-50' : ''}`} style={{ transition: 'opacity 0.2s' }}>
            <div className="image-card">
              <img src={`data:image/png;base64,${currentItem.pair.image1.data}`} alt="Image 1" />
            </div>
            <div className="image-card">
              <img src={`data:image/png;base64,${currentItem.pair.image2.data}`} alt="Image 2" />
            </div>
          </div>

          <div className="buttons">
            <button 
              className={`btn-yes ${currentItem.vote === true ? 'selected' : ''}`}
              onClick={() => handleVote(true)}
            >
              YES (J)
            </button>
            <button 
              className={`btn-no ${currentItem.vote === false ? 'selected' : ''}`}
              onClick={() => handleVote(false)}
            >
              NO (I)
            </button>
          </div>

          <div className="status">
            Pair {currentIndex + 1} / {history.length} • Strategy: {currentItem.pair.debug_strategy}
          </div>
          
          <div className="shortcut-hint">
            Use <kbd>J</kbd> / <kbd>I</kbd> to vote • <kbd>←</kbd> / <kbd>→</kbd> to navigate
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
