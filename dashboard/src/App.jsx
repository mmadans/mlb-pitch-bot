import React, { useState, useEffect } from 'react';
import { Target, Zap, Activity, Users, ChevronRight, BarChart3, Database } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Fetch the exported json data from the public folder
    fetch('/data/dashboard_data.json')
      .then(res => res.json())
      .then(json => {
        setData(json);
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load dashboard data:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-900 text-white">
        <div className="animate-spin mr-3">
          <Activity size={24} className="text-blue-500" />
        </div>
        <p className="text-xl font-light">Loading Pitch Insights...</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-900 text-white">
        <p className="text-red-400">Error loading dashboard data. Make sure to run the export script.</p>
      </div>
    );
  }

  // Prepare Chart Data
  const pieData = Object.keys(data.pitch_distribution).map(key => ({
    name: key,
    value: data.pitch_distribution[key]
  }));
  
  const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b'];

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 p-6 md:p-8 font-sans selection:bg-blue-500/30">
      
      {/* Header */}
      <header className="mb-10 flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h1 className="text-4xl font-extrabold tracking-tight flex items-center gap-3">
            <Target className="text-blue-500" size={36} />
            <span className="gradient-text">MLB Pitch Bot</span>
          </h1>
          <p className="text-slate-400 mt-2 font-medium">Advanced Surprisal & Tendency Analytics</p>
        </div>
        
        <div className="glass-panel px-4 py-2 rounded-full text-xs font-semibold text-slate-300 flex items-center gap-2 border border-slate-700/50">
          <Database size={14} className="text-emerald-400" />
          Last Updated: {new Date(data.last_updated).toLocaleString()}
        </div>
      </header>

      {/* Overview Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="glass-panel p-6 rounded-2xl relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Activity size={64} />
          </div>
          <p className="text-slate-400 text-sm font-semibold uppercase tracking-wider mb-2">Total Pitches Tracked</p>
          <p className="text-4xl font-bold text-white">{data.summary.total_pitches.toLocaleString()}</p>
        </div>
        
        <div className="glass-panel p-6 rounded-2xl relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Users size={64} />
          </div>
          <p className="text-slate-400 text-sm font-semibold uppercase tracking-wider mb-2">Unique Games</p>
          <p className="text-4xl font-bold text-white">{data.summary.total_games.toLocaleString()}</p>
        </div>

        <div className="glass-panel p-6 rounded-2xl relative overflow-hidden group bg-gradient-to-br from-slate-800/80 to-blue-900/30">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Zap size={64} className="text-blue-400" />
          </div>
          <p className="text-blue-300 text-sm font-semibold uppercase tracking-wider mb-2">Strikeouts Logged</p>
          <p className="text-4xl font-bold text-white">{data.summary.total_strikeouts.toLocaleString()}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
        
        {/* Surprise List */}
        <div className="lg:col-span-2 glass-panel rounded-2xl p-1 overflow-hidden">
          <div className="p-5 border-b border-slate-700/50 flex justify-between items-center">
            <h2 className="text-xl font-bold flex items-center gap-2">
              <Zap className="text-amber-400" size={20} />
              Most Surprising Strikeouts
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-slate-800/50 text-slate-400 text-xs uppercase tracking-wider">
                  <th className="p-4 font-semibold">Matchup</th>
                  <th className="p-4 font-semibold text-center">Count</th>
                  <th className="p-4 font-semibold">Pitch Thrown</th>
                  <th className="p-4 font-semibold text-right">Surprisal Score</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/50">
                {data.top_surprises.map((pitch, i) => (
                  <tr key={i} className="hover:bg-slate-800/30 transition-colors group cursor-default">
                    <td className="p-4">
                      <div className="font-medium text-white group-hover:text-blue-300 transition-colors">{pitch.pitcher}</div>
                      <div className="text-xs text-slate-500">vs {pitch.batter_id} • {pitch.inning}</div>
                    </td>
                    <td className="p-4 text-center">
                      <span className="bg-slate-800 px-2 py-1 rounded font-mono text-sm border border-slate-700">{pitch.count}</span>
                    </td>
                    <td className="p-4">
                      <div className="font-medium">{pitch.pitch_type}</div>
                      <div className="text-xs text-slate-400">{pitch.speed} mph</div>
                    </td>
                    <td className="p-4 text-right">
                      <div className="inline-flex items-center justify-end gap-2">
                        <div className="text-xs text-slate-500">{(pitch.prob * 100).toFixed(1)}% prob</div>
                        <div className="bg-blue-500/20 text-blue-300 border border-blue-500/30 px-3 py-1 rounded-full font-bold text-sm">
                          {pitch.surprisal} bits
                        </div>
                      </div>
                    </td>
                  </tr>
                ))}
                {data.top_surprises.length === 0 && (
                  <tr>
                    <td colSpan="4" className="p-8 text-center text-slate-500 italic">No surprise pitches logged yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

        {/* Charts Side Panel */}
        <div className="flex flex-col gap-8">
          
          <div className="glass-panel p-6 rounded-2xl">
             <h2 className="text-lg font-bold mb-4 flex items-center gap-2">
              <BarChart3 className="text-purple-400" size={18} />
              Pitch Type Distribution
            </h2>
            <div className="h-64 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={60}
                    outerRadius={80}
                    paddingAngle={5}
                    dataKey="value"
                    stroke="rgba(0,0,0,0)"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <RechartsTooltip 
                    contentStyle={{ backgroundColor: '#1e293b', borderColor: '#334155', borderRadius: '8px', color: '#fff' }}
                    itemStyle={{ color: '#fff' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="grid grid-cols-2 gap-2 mt-2">
              {pieData.map((entry, index) => (
                <div key={entry.name} className="flex items-center gap-2 text-sm text-slate-300">
                  <div className="w-3 h-3 rounded-full" style={{ backgroundColor: COLORS[index % COLORS.length] }}></div>
                  <span>{entry.name}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="glass-panel p-6 rounded-2xl flex-grow">
            <h2 className="text-lg font-bold mb-4 text-slate-200">Top Pitchers Tracked</h2>
            <ul className="space-y-3">
              {data.top_pitchers.slice(0, 5).map((p, i) => (
                <li key={i} className="flex justify-between items-center bg-slate-800/40 p-3 rounded-xl border border-slate-700/30">
                  <span className="font-medium text-slate-200">{p.name}</span>
                  <span className="text-xs font-mono bg-slate-900 border border-slate-700 px-2 py-1 rounded text-slate-400">{p.count} pitches</span>
                </li>
              ))}
            </ul>
          </div>

        </div>

      </div>
    </div>
  );
}

export default App;
