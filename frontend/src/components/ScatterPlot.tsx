import { useEffect, useRef } from 'react';
import createScatterplot from 'regl-scatterplot';

interface Point {
  id: string;
  x: number;
  y: number;
  label?: string | null;
  predicted_label?: string | null;
  confidence?: number;
}

interface ScatterPlotProps {
  points: Point[];
  currentImageId?: string;
  onPointClick?: (id: string) => void;
}

// Modern, distinct palette for 10 classes (0-9)
const LABEL_COLORS: Record<string, [number, number, number, number]> = {
  "zero": [0.94, 0.33, 0.33, 1.0],   // Red
  "one": [0.94, 0.6, 0.25, 1.0],    // Orange
  "two": [0.98, 0.85, 0.37, 1.0],   // Yellow
  "three": [0.38, 0.87, 0.54, 1.0], // Green
  "four": [0.27, 0.8, 0.82, 1.0],   // Teal
  "five": [0.33, 0.62, 0.94, 1.0],  // Blue
  "six": [0.55, 0.45, 0.96, 1.0],   // Indigo
  "seven": [0.8, 0.47, 0.96, 1.0],  // Purple
  "eight": [0.94, 0.38, 0.72, 1.0], // Pink
  "nine": [0.62, 0.65, 0.69, 1.0]   // Grayish
};

// Brighter gray for better visibility, lower alpha to not overwhelm
const UNLABELLED_COLOR: [number, number, number, number] = [0.5, 0.5, 0.5, 0.5];
const CURRENT_COLOR: [number, number, number, number] = [1.0, 1.0, 1.0, 1.0]; // Bright White

export function ScatterPlot({ points, currentImageId, onPointClick: _onPointClick }: ScatterPlotProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const scatterplotRef = useRef<any>(null);
  const prevPointCountRef = useRef<number>(0);

  useEffect(() => {
    if (!canvasRef.current) return;

    const canvas = canvasRef.current;
    const { width, height } = canvas.getBoundingClientRect();
    
    const scatterplot = createScatterplot({
      canvas,
      width,
      height,
      pointSize: 6, // Increased from 4
      pointColor: UNLABELLED_COLOR, 
      showReticle: false, 
      backgroundColor: [0.05, 0.05, 0.07, 1.0], // Slightly lighter than pure black for contrast
    });

    scatterplotRef.current = scatterplot;

    // @ts-ignore
    scatterplot.subscribe('pointover', () => {
      canvas.style.cursor = 'crosshair';
    });
    
    // @ts-ignore
    scatterplot.subscribe('pointout', () => {
      canvas.style.cursor = 'default';
    });

    return () => {
      scatterplot.destroy();
    };
  }, []);

  useEffect(() => {
    if (!scatterplotRef.current || points.length === 0) return;

    // Sort points for proper z-indexing:
    // 1. Unlabelled (Background)
    // 2. Labelled (Midground)
    // 3. Current (Foreground)
    const sortedPoints = [...points].sort((a, b) => {
        const aIsCurrent = a.id === currentImageId;
        const bIsCurrent = b.id === currentImageId;
        if (aIsCurrent) return 1;
        if (bIsCurrent) return -1;
        
        const aLabelled = !!a.label;
        const bLabelled = !!b.label;
        if (aLabelled && !bLabelled) return 1;
        if (!aLabelled && bLabelled) return -1;
        
        return 0;
    });

    const data = sortedPoints.map(p => [p.x, p.y]);
    
    const colors = sortedPoints.map(p => {
      if (p.id === currentImageId) return CURRENT_COLOR;

      // User-labeled points: full color
      if (p.label && LABEL_COLORS[p.label.toLowerCase()]) {
        return LABEL_COLORS[p.label.toLowerCase()];
      } else if (p.label) {
        return [0.4, 0.8, 0.4, 1.0];
      }

      // Predicted labels: color with confidence-based alpha
      if (p.predicted_label && LABEL_COLORS[p.predicted_label.toLowerCase()]) {
        const baseColor = LABEL_COLORS[p.predicted_label.toLowerCase()];
        const confidence = p.confidence ?? 0;
        // Alpha ranges from 0.15 (low confidence) to 0.7 (high confidence)
        const alpha = 0.15 + confidence * 0.55;
        return [baseColor[0], baseColor[1], baseColor[2], alpha] as [number, number, number, number];
      }

      return UNLABELLED_COLOR;
    });

    const sizes = sortedPoints.map(p => {
        if (p.id === currentImageId) return 25; // Massive highlight
        if (p.label) return 8; // User-labeled points are larger

        // Predicted points: size scales with confidence
        if (p.predicted_label && p.confidence) {
          // Size ranges from 4 (low confidence) to 7 (high confidence)
          return 4 + p.confidence * 3;
        }

        return 4; // Unlabeled, no prediction
    });

    // Only use transitions when point count is the same (regl-scatterplot requirement)
    const canTransition = prevPointCountRef.current === points.length && prevPointCountRef.current > 0;
    const isFirstDraw = prevPointCountRef.current === 0;
    prevPointCountRef.current = points.length;

    scatterplotRef.current.draw(data, {
        pointColor: colors,
        pointSize: sizes,
        transition: canTransition,
        transitionDuration: canTransition ? 600 : 0,
        transitionEasing: 'cubicOut'
    });

    // Auto-fit view on first draw by calculating bounds
    if (isFirstDraw && data.length > 0) {
      const xs = data.map(d => d[0]);
      const ys = data.map(d => d[1]);
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const minY = Math.min(...ys);
      const maxY = Math.max(...ys);
      const padX = (maxX - minX) * 0.1;
      const padY = (maxY - minY) * 0.1;

      scatterplotRef.current.zoomToArea({
        x: minX - padX,
        y: minY - padY,
        width: (maxX - minX) + padX * 2,
        height: (maxY - minY) + padY * 2
      }, { transition: false });
    }
    
  }, [points, currentImageId]);

  return (
    <div className="w-full h-full relative overflow-hidden bg-[#0a0a0c] rounded-xl shadow-inner border border-white/5">
      <canvas 
        ref={canvasRef} 
        className="w-full h-full block"
        style={{ width: '100%', height: '100%' }}
      />
      {/* Legend Overlay */}
      <div className="absolute bottom-4 left-4 pointer-events-none flex flex-wrap gap-2 max-w-[90%] opacity-90">
         {Object.entries(LABEL_COLORS).map(([label, rgba]) => (
             <div key={label} className="flex items-center gap-1.5 bg-black/60 backdrop-blur-md px-2 py-1 rounded-md border border-white/10 shadow-sm">
                 <div className="w-2.5 h-2.5 rounded-full shadow-[0_0_4px_rgba(0,0,0,0.5)]" style={{ backgroundColor: `rgba(${rgba[0]*255}, ${rgba[1]*255}, ${rgba[2]*255}, 1)` }} />
                 <span className="text-[10px] uppercase text-white font-bold tracking-wider">{label}</span>
             </div>
         ))}
      </div>
    </div>
  );
}
