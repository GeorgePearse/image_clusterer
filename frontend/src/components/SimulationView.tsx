import { useState, useEffect } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { API_URL } from "../api";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface SimulationResult {
  id: string;
  status: string;
  request: {
    strategy: string;
    num_images: number;
    max_labels: number;
  };
  start_time: number;
  duration?: number;
  results?: {
    labels_history: {
      step: number;
      accuracy: number;
      cluster_purity: number;
      num_labeled: number;
    }[];
    final_state: any;
  };
  error?: string;
}

export function SimulationView() {
  const [strategies, setStrategies] = useState<string[]>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<string>("cluster_chain");
  const [numImages, setNumImages] = useState<number>(1000);
  const [maxLabels, setMaxLabels] = useState<number>(200);
  const [simulations, setSimulations] = useState<SimulationResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    fetchStrategies();
    fetchResults();
    const interval = setInterval(fetchResults, 2000);
    return () => clearInterval(interval);
  }, []);

  const fetchStrategies = async () => {
    try {
      const res = await fetch(`${API_URL}/simulation/strategies`);
      const data = await res.json();
      setStrategies(data);
      if (data.length > 0) setSelectedStrategy(data[0]);
    } catch (e) {
      console.error("Failed to fetch strategies", e);
    }
  };

  const fetchResults = async () => {
    try {
      const res = await fetch(`${API_URL}/simulation/results`);
      const data = await res.json();
      setSimulations(data);
    } catch (e) {
      console.error("Failed to fetch results", e);
    }
  };

  const runSimulation = async () => {
    setIsLoading(true);
    try {
      await fetch(`${API_URL}/simulation/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          strategy: selectedStrategy,
          num_images: numImages,
          max_labels: maxLabels,
        }),
      });
      fetchResults();
    } catch (e) {
      console.error("Failed to start simulation", e);
    } finally {
      setIsLoading(false);
    }
  };

  // Prepare chart data: combine all completed simulations
  // We want to compare Accuracy vs Num Labeled
  // This is tricky if x-values differ. We can just plot lines for each sim.
  
  // Create a color map for strategies
  const colors = ["#8884d8", "#82ca9d", "#ffc658", "#ff7300", "#0088fe", "#00C49F"];
  
  return (
    <div className="h-full flex flex-col gap-6 p-6 overflow-hidden">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold tracking-tight">Active Learning Simulation</h2>
        <div className="flex items-center gap-4">
          <select 
            className="h-9 w-[180px] rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
            value={selectedStrategy}
            onChange={(e) => setSelectedStrategy(e.target.value)}
          >
            {strategies.map(s => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Images:</span>
            <Input 
              type="number" 
              className="w-20 h-9" 
              value={numImages} 
              onChange={(e) => setNumImages(parseInt(e.target.value))}
            />
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-muted-foreground">Labels:</span>
            <Input 
              type="number" 
              className="w-20 h-9" 
              value={maxLabels} 
              onChange={(e) => setMaxLabels(parseInt(e.target.value))}
            />
          </div>
          <Button onClick={runSimulation} disabled={isLoading}>
            {isLoading ? "Starting..." : "Run Simulation"}
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 h-full min-h-0">
        {/* Results List */}
        <Card className="col-span-1 p-4 flex flex-col gap-4 overflow-hidden">
          <h3 className="font-semibold">History</h3>
          <div className="overflow-y-auto flex-1 flex flex-col gap-2">
            {simulations.map((sim) => (
              <div key={sim.id} className="p-3 border rounded-lg bg-card/50 text-sm">
                <div className="flex justify-between items-start mb-1">
                  <span className="font-medium capitalize">{sim.request.strategy}</span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded-full uppercase ${
                    sim.status === 'done' ? 'bg-green-500/20 text-green-500' :
                    sim.status === 'error' ? 'bg-red-500/20 text-red-500' :
                    'bg-yellow-500/20 text-yellow-500'
                  }`}>
                    {sim.status}
                  </span>
                </div>
                <div className="text-xs text-muted-foreground grid grid-cols-2 gap-1">
                  <span>Images: {sim.request.num_images}</span>
                  <span>Max Labels: {sim.request.max_labels}</span>
                  {sim.results && (
                    <>
                      <span>Duration: {sim.duration?.toFixed(1)}s</span>
                      <span>Final Acc: {(sim.results.final_state.accuracy * 100).toFixed(1)}%</span>
                      <span>KNN Acc: {(sim.results.final_state.knn_accuracy * 100).toFixed(1)}%</span>
                    </>
                  )}
                </div>
                {sim.error && <div className="mt-2 text-xs text-red-400">{sim.error}</div>}
              </div>
            ))}
            {simulations.length === 0 && (
              <div className="text-center text-muted-foreground py-8">No simulations run yet.</div>
            )}
          </div>
        </Card>

        {/* Charts */}
        <Card className="col-span-2 p-4 flex flex-col min-h-0">
          <h3 className="font-semibold mb-4">Model Performance (KNN Accuracy)</h3>
          <div className="flex-1 min-h-0">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#333" />
                <XAxis 
                  dataKey="num_labeled" 
                  type="number" 
                  domain={[0, 'dataMax']} 
                  label={{ value: 'Labels', position: 'insideBottomRight', offset: -5 }} 
                />
                <YAxis domain={[0, 1]} label={{ value: 'Accuracy', angle: -90, position: 'insideLeft' }} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f1f1f', border: 'none' }}
                  itemStyle={{ fontSize: '12px' }}
                />
                <Legend />
                {simulations
                  .filter(s => s.status === 'done' && s.results)
                  .map((sim, idx) => (
                    <Line
                      key={sim.id}
                      data={sim.results!.labels_history}
                      type="monotone"
                      dataKey="knn_accuracy"
                      name={`${sim.request.strategy} (${sim.request.num_images})`}
                      stroke={colors[idx % colors.length]}
                      dot={false}
                      strokeWidth={2}
                    />
                  ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      </div>
    </div>
  );
}
