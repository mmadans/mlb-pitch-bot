import React, { useState, useEffect } from 'react';
import { Target, Zap, Activity, Users, ChevronRight, BarChart3, Database, TrendingUp, Info, Calendar, Play, ChevronLeft, MapPin } from 'lucide-react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip, BarChart, Bar, XAxis, YAxis, CartesianGrid } from 'recharts';

function App() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeView, setActiveView] = useState('dashboard');
  
  // Dashboard State
  const [selectedPitcher, setSelectedPitcher] = useState(null);
  
  // Explorer State
  const [selectedDate, setSelectedDate] = useState(null);
  const [selectedGame, setSelectedGame] = useState(null);
  const [selectedInning, setSelectedInning] = useState('1');
  const [selectedHalf, setSelectedHalf] = useState('top');
  const [selectedAB, setSelectedAB] = useState(null);

  useEffect(() => {
    fetch('/data/dashboard_data.json')
      .then(res => res.json())
      .then(json => {
        setData(json);
        if (json.pitcher_spotlight && json.pitcher_spotlight.length > 0) {
          setSelectedPitcher(json.pitcher_spotlight[0]);
        }
        
        // Init explorer if data exists
        if (json.explorer) {
          const dates = Object.keys(json.explorer).sort((a,b) => b.localeCompare(a));
          if (dates.length > 0) {
            setSelectedDate(dates[0]);
            const games = Object.keys(json.explorer[dates[0]].games);
            if (games.length > 0) {
              setSelectedGame(json.explorer[dates[0]].games[games[0]]);
            }
          }
        }
        
        setLoading(false);
      })
      .catch(err => {
        console.error("Failed to load dashboard data:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center bg-[#0a0f18] text-white">
        <Activity size={48} className="text-blue-500 animate-[pulse_2s_infinite]" />
        <p className="text-2xl font-light mt-8 tracking-widest text-slate-400 uppercase">Analyzing Neural Data...</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-[#0a0f18] text-white p-8 text-center">
        <div className="glass-panel p-12 rounded-3xl border border-red-500/20 max-w-md">
           <Info className="text-red-400 mx-auto mb-4" size={48} />
           <p className="text-xl font-medium">Data Integrity Error</p>
           <p className="text-slate-400 mt-2">Could not synchronize with the pitch database.</p>
        </div>
      </div>
    );
  }

  const COLORS = ['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b', '#ef4444'];
  
  // Dashboard Calculations
  const pieData = Object.keys(data.pitch_distribution).map(key => ({
    name: key,
    value: data.pitch_distribution[key]
  }));

  const spotlightData = selectedPitcher ? Object.keys(selectedPitcher.distribution).map(key => ({
    name: key,
    count: selectedPitcher.distribution[key],
    avgSpeed: selectedPitcher.avg_speeds[key] || 0
  })) : [];

  // Explorer Calculations
  const dates = data.explorer ? Object.keys(data.explorer).sort((a,b) => b.localeCompare(a)) : [];
  const gamesOnDate = (data.explorer && selectedDate) ? Object.values(data.explorer[selectedDate].games) : [];
  const activeInningData = selectedGame?.innings[selectedInning] || { top: [], bottom: [] };
  const matchups = activeInningData[selectedHalf] || [];

  return (
    <div className="min-h-screen bg-[#070b13] text-slate-100 font-sans selection:bg-blue-500/30">
      
      {/* Background Decor */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-600/10 blur-[120px] rounded-full"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-purple-600/10 blur-[120px] rounded-full"></div>
      </div>

      <div className="relative z-10 max-w-7xl mx-auto p-4 md:p-8">
        
        {/* Navigation & Header */}
        <header className="mb-12 flex flex-col lg:flex-row justify-between items-start lg:items-center gap-8">
          <div className="animate-in fade-in slide-in-from-left duration-700">
            <h1 className="text-4xl font-black tracking-tighter flex items-center gap-4">
              <span className="bg-blue-600 p-2 rounded-xl">
                 <Target className="text-white" size={28} />
              </span>
              <span className="bg-gradient-to-r from-white via-slate-200 to-slate-400 bg-clip-text text-transparent italic">
                PITCH BOT <span className="text-blue-500 not-italic">INSIGHTS</span>
              </span>
            </h1>
          </div>
          
          <nav className="flex items-center gap-2 p-1 bg-slate-900/50 backdrop-blur-xl border border-white/5 rounded-2xl">
            <button 
              onClick={() => setActiveView('dashboard')}
              className={`px-6 py-2 rounded-xl text-xs font-black tracking-widest uppercase transition-all duration-300 ${activeView === 'dashboard' ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' : 'text-slate-500 hover:text-slate-300'}`}
            >
              Overview
            </button>
            <button 
              onClick={() => setActiveView('explorer')}
              className={`px-6 py-2 rounded-xl text-xs font-black tracking-widest uppercase transition-all duration-300 ${activeView === 'explorer' ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20' : 'text-slate-500 hover:text-slate-300'}`}
            >
              Matchup Explorer
            </button>
          </nav>

          <div className="hidden lg:flex flex-col items-end gap-1">
             <div className="glass-panel px-4 py-2 rounded-xl text-[10px] font-bold text-slate-400 flex items-center gap-3 border border-emerald-500/10 uppercase tracking-wider bg-emerald-500/[0.02]">
                <span className="flex h-2 w-2 rounded-full bg-emerald-500 animate-pulse"></span>
                LIVE ENGINE: v2.4
             </div>
          </div>
        </header>

        {activeView === 'dashboard' ? (
          /* DASHBOARD VIEW */
          <div className="animate-in fade-in slide-in-from-bottom duration-700">
            {/* Overview Stats */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
              {[
                { label: 'Pitches Tracked', value: data.summary.total_pitches, icon: <Activity className="text-blue-400" />, color: 'from-blue-500/10' },
                { label: 'Unique Games', value: data.summary.total_games, icon: <Database className="text-purple-400" />, color: 'from-purple-500/10' },
                { label: 'Strikeouts Logged', value: data.summary.total_strikeouts, icon: <Zap className="text-amber-400" />, color: 'from-amber-500/10' }
              ].map((stat, i) => (
                <div key={i} className={`glass-panel p-8 rounded-[2rem] border border-white/5 bg-gradient-to-br ${stat.color} to-transparent group transition-all duration-500 hover:border-white/10 hover:translate-y-[-4px]`}>
                  <div className="flex items-center justify-between mb-4">
                     <div className="bg-slate-900/50 p-3 rounded-2xl border border-white/5">{stat.icon}</div>
                     <TrendingUp size={16} className="text-slate-600 group-hover:text-slate-400 transition-colors" />
                  </div>
                  <p className="text-slate-500 text-xs font-bold uppercase tracking-widest">{stat.label}</p>
                  <p className="text-4xl font-black text-white mt-1 tabular-nums">{stat.value.toLocaleString()}</p>
                </div>
              ))}
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              
              {/* Main Content Area */}
              <div className="lg:col-span-2 space-y-8">
                
                {/* Surprise Explorer */}
                <div className="glass-panel rounded-[2.5rem] border border-white/5 overflow-hidden">
                   <div className="p-8 border-b border-white/5 flex justify-between items-center bg-white/2">
                      <h2 className="text-2xl font-black italic tracking-tight flex items-center gap-3">
                        <Zap className="text-amber-400 fill-amber-400/20" size={24} />
                        TOP SURPRISAL EVENTS
                      </h2>
                   </div>
                   
                   <div className="overflow-x-auto">
                     <table className="w-full text-left border-collapse">
                       <thead>
                         <tr className="bg-slate-900/40 text-slate-500 text-[10px] uppercase font-black tracking-[0.2em] border-b border-white/5">
                           <th className="px-8 py-4">Matchup</th>
                           <th className="px-8 py-4">Context</th>
                           <th className="px-8 py-4">Pitch Profile</th>
                           <th className="px-8 py-4 text-right">Model Metrics</th>
                         </tr>
                       </thead>
                       <tbody className="divide-y divide-white/[0.02]">
                         {data.top_surprises.map((pitch, i) => (
                           <tr key={i} className="hover:bg-blue-500/[0.02] transition-colors group">
                             <td className="px-8 py-6">
                               <div className="font-bold text-white group-hover:text-blue-400 transition-colors">{pitch.pitcher}</div>
                               <div className="text-[10px] text-slate-500 font-bold uppercase mt-0.5 tracking-wider">vs {pitch.batter}</div>
                             </td>
                             <td className="px-8 py-6">
                               <div className="inline-flex items-center bg-slate-900 border border-white/5 px-2 py-1 rounded text-[10px] font-bold text-slate-400 tracking-tighter">
                                 {pitch.count} • {pitch.inning}
                               </div>
                             </td>
                             <td className="px-8 py-6">
                               <div className="flex items-center gap-3">
                                  <span className={`w-1.5 h-1.5 rounded-full ${pitch.outcome === 'Strikeout' ? 'bg-amber-400' : 'bg-red-500'}`}></span>
                                  <div>
                                    <div className="font-bold text-sm">{pitch.pitch_type}</div>
                                    <div className="text-[10px] text-slate-500 font-medium">{pitch.speed} MPH • {pitch.pitch_family}</div>
                                  </div>
                               </div>
                             </td>
                             <td className="px-8 py-6 text-right">
                               <div className="flex flex-col items-end gap-1">
                                 <div className="text-[10px] font-bold text-slate-500">P(pitch) = {(pitch.prob * 100).toFixed(2)}%</div>
                                 <div className="px-3 py-1 bg-blue-600/20 border border-blue-500/30 rounded-lg text-blue-400 font-black text-xs">
                                    {pitch.surprisal.toFixed(2)} <span className="text-[8px] opacity-60 font-medium">BITS</span>
                                 </div>
                               </div>
                             </td>
                           </tr>
                         ))}
                       </tbody>
                     </table>
                   </div>
                </div>

                {/* Pitcher Spotlight */}
                <div className="glass-panel rounded-[2.5rem] border border-white/5 p-8 relative overflow-hidden">
                    <div className="absolute top-0 right-0 p-12 opacity-5 pointer-events-none">
                        <Users size={120} />
                    </div>
                    
                    <h2 className="text-2xl font-black italic tracking-tight mb-8 flex items-center gap-3">
                      <Users className="text-purple-400" size={24} />
                      PITCHER SPOTLIGHT
                    </h2>

                    <div className="flex flex-wrap gap-2 mb-8">
                      {data.pitcher_spotlight.map((p) => (
                        <button 
                          key={p.name}
                          onClick={() => setSelectedPitcher(p)}
                          className={`px-4 py-2 rounded-xl text-xs font-bold transition-all duration-300 border ${
                            selectedPitcher?.name === p.name 
                            ? 'bg-blue-600 text-white border-blue-500 shadow-lg shadow-blue-900/40' 
                            : 'bg-slate-900 text-slate-500 border-white/5 hover:border-white/10'
                          }`}
                        >
                          {p.name}
                        </button>
                      ))}
                    </div>

                    {selectedPitcher && (
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-12 animate-in fade-in slide-in-from-bottom duration-500">
                        <div>
                            <div className="flex items-end justify-between mb-6">
                               <div>
                                  <p className="text-[10px] font-black text-blue-500 uppercase tracking-[0.2em] mb-1">Featured Athlete</p>
                                  <h3 className="text-3xl font-black text-white">{selectedPitcher.name}</h3>
                               </div>
                               <div className="text-right">
                                  <p className="text-2xl font-black text-white">{selectedPitcher.total_pitches}</p>
                                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Pitches in DB</p>
                               </div>
                            </div>
                            
                            <div className="space-y-4">
                              {spotlightData.map((item, i) => (
                                <div key={item.name} className="bg-white/2 border border-white/5 p-4 rounded-2xl flex items-center justify-between group hover:bg-white/5 transition-colors">
                                   <div className="flex items-center gap-3">
                                      <div className="w-2 h-2 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }}></div>
                                      <div>
                                         <p className="text-xs font-bold text-white uppercase">{item.name}</p>
                                         <p className="text-[10px] text-slate-500">{item.avgSpeed} MPH Avg</p>
                                      </div>
                                   </div>
                                   <div className="text-right">
                                      <p className="text-xs font-black text-slate-200">{Math.round((item.count / selectedPitcher.total_pitches) * 100)}%</p>
                                      <div className="w-16 h-1 bg-slate-800 rounded-full mt-1 overflow-hidden">
                                         <div 
                                            className="h-full bg-blue-500 rounded-full" 
                                            style={{ width: `${(item.count / selectedPitcher.total_pitches) * 100}%` }}
                                          ></div>
                                      </div>
                                   </div>
                                </div>
                              ))}
                            </div>
                        </div>
                        
                        <div className="h-64 bg-slate-900/50 rounded-3xl p-6 border border-white/5">
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={spotlightData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                                <XAxis 
                                  dataKey="name" 
                                  axisLine={false} 
                                  tickLine={false} 
                                  tick={{ fill: '#64748b', fontSize: 10, fontWeight: 700 }} 
                                />
                                <YAxis hide />
                                <Bar 
                                  dataKey="count" 
                                  radius={[6, 6, 0, 0]} 
                                  fill="#3b82f6"
                                />
                              </BarChart>
                            </ResponsiveContainer>
                        </div>
                      </div>
                    )}
                </div>

              </div>

              {/* Sidebar */}
              <div className="space-y-8">
                
                {/* Global Distribution */}
                <div className="glass-panel p-8 rounded-[2.5rem] border border-white/5 relative overflow-hidden">
                     <h2 className="text-lg font-black italic tracking-tight mb-8 flex items-center gap-2">
                      <BarChart3 className="text-blue-400" size={20} />
                      LEAGUE PROFILE
                    </h2>
                    
                    <div className="h-64 mb-8">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie
                            data={pieData}
                            cx="50%"
                            cy="50%"
                            innerRadius={65}
                            outerRadius={85}
                            paddingAngle={8}
                            dataKey="value"
                            stroke="rgba(0,0,0,0)"
                          >
                            {pieData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} cornerRadius={8} />
                            ))}
                          </Pie>
                        </PieChart>
                      </ResponsiveContainer>
                      
                      <div className="absolute top-[50%] left-[50%] translate-x-[-50%] translate-y-[-10%] text-center pointer-events-none">
                         <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest leading-none">Family</p>
                         <p className="text-xl font-black text-white">Mix</p>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 gap-3">
                      {pieData.map((item, i) => (
                        <div key={item.name} className="flex items-center justify-between p-3 rounded-2xl bg-white/[0.03] border border-white/5">
                            <div className="flex items-center gap-3">
                               <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: COLORS[i % COLORS.length] }}></div>
                               <span className="text-[10px] font-black text-slate-400 uppercase tracking-widest">{item.name}</span>
                            </div>
                            <span className="text-xs font-black text-white">{Math.round((item.value / data.summary.total_pitches) * 100)}%</span>
                        </div>
                      ))}
                    </div>
                </div>

                {/* Top Pitchers Table */}
                <div className="glass-panel p-8 rounded-[2.5rem] border border-white/5">
                   <h2 className="text-lg font-black italic tracking-tight mb-6">MOST TRACKED</h2>
                   <div className="space-y-3">
                      {data.top_pitchers.slice(0, 6).map((p, i) => (
                        <div key={i} className="flex justify-between items-center bg-slate-900 border border-white/5 p-4 rounded-2xl group cursor-pointer hover:border-blue-500/30 transition-all duration-300">
                          <div className="flex items-center gap-4">
                             <span className="text-[10px] font-black text-slate-700">0{i+1}</span>
                             <span className="font-bold text-sm text-slate-300 group-hover:text-white transition-colors">{p.name}</span>
                          </div>
                          <div className="text-right">
                             <span className="text-[10px] font-black bg-blue-600/10 border border-blue-500/20 px-2.5 py-1 rounded-lg text-blue-400">
                               {p.count} <span className="text-[8px] opacity-60">PITCHES</span>
                             </span>
                          </div>
                        </div>
                      ))}
                   </div>
                </div>

              </div>

            </div>
          </div>
        ) : (
          /* MATCHUP EXPLORER VIEW */
          <div className="animate-in fade-in slide-in-from-bottom duration-700 space-y-8">
            
            {/* Explorer Toolbar */}
            <div className="glass-panel p-6 rounded-3xl border border-white/5 space-y-6">
                
                {/* 1. Date Selector */}
                <div className="flex flex-col gap-3">
                   <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest flex items-center gap-2">
                     <Calendar size={12} /> Select Game Date
                   </p>
                   <div className="flex gap-2 overflow-x-auto pb-2 scrollbar-none">
                     {dates.map((date) => (
                       <button
                         key={date}
                         onClick={() => {
                           setSelectedDate(date);
                           const firstGame = Object.values(data.explorer[date].games)[0];
                           setSelectedGame(firstGame);
                           setSelectedAB(null);
                         }}
                         className={`px-5 py-2.5 rounded-xl text-xs font-bold whitespace-nowrap transition-all border ${
                           selectedDate === date 
                           ? 'bg-blue-600 border-blue-400 text-white shadow-xl shadow-blue-900/40' 
                           : 'bg-slate-900 border-white/5 text-slate-500 hover:border-white/10'
                         }`}
                       >
                         {new Date(date).toLocaleDateString('en-US', { month: 'short', day: 'numeric', weekday: 'short' })}
                       </button>
                     ))}
                   </div>
                </div>

                {/* 2. Game Selector */}
                {selectedDate && (
                  <div className="flex flex-col gap-3">
                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest flex items-center gap-2">
                      <Play size={12} /> Active Games
                    </p>
                    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
                       {gamesOnDate.map((game) => (
                         <button
                           key={game.id}
                           onClick={() => {
                             setSelectedGame(game);
                             setSelectedAB(null);
                           }}
                           className={`p-4 rounded-2xl border text-left transition-all ${
                             selectedGame?.id === game.id
                             ? 'bg-gradient-to-br from-blue-600 to-blue-700 border-blue-400 shadow-lg shadow-blue-900/30 scale-[1.02]'
                             : 'bg-slate-900 border-white/5 hover:border-white/10'
                           }`}
                         >
                            <div className="flex justify-between items-start mb-2">
                               <span className={`text-[8px] font-black px-1.5 py-0.5 rounded ${selectedGame?.id === game.id ? 'bg-white/20 text-white' : 'bg-blue-500/10 text-blue-400 uppercase'}`}>
                                 #{game.id.slice(-6)}
                               </span>
                               <MapPin size={12} className={selectedGame?.id === game.id ? 'text-white/60' : 'text-slate-600'} />
                            </div>
                            <div className={`font-black uppercase tracking-tight ${selectedGame?.id === game.id ? 'text-white' : 'text-slate-300'}`}>
                               {game.matchup}
                            </div>
                            <div className={`text-[10px] font-bold ${selectedGame?.id === game.id ? 'text-white/60' : 'text-slate-600'}`}>
                               Venue: {game.venue}
                            </div>
                         </button>
                       ))}
                    </div>
                  </div>
                )}
            </div>

            {/* Inning & Matchup Explorer */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              
              {/* Sidebar Left: Inning Nav */}
              <div className="lg:col-span-3 space-y-4">
                 <div className="glass-panel p-6 rounded-3xl border border-white/5 overflow-hidden">
                    <h3 className="text-xs font-black text-slate-400 uppercase tracking-[0.2em] mb-6 flex items-center gap-2">
                       Inning Layout
                    </h3>
                    <div className="grid grid-cols-3 gap-2 mb-6">
                       {['1','2','3','4','5','6','7','8','9','10'].map(inn => (
                         <button 
                            key={inn}
                            onClick={() => { setSelectedInning(inn); setSelectedAB(null); }}
                            className={`h-10 rounded-lg text-xs font-black border transition-all ${
                              selectedInning === inn 
                              ? 'bg-blue-600 border-blue-400 text-white shadow-lg' 
                              : 'bg-slate-800/50 border-white/5 text-slate-500 hover:text-slate-300'
                            }`}
                         >
                           {inn}
                         </button>
                       ))}
                    </div>
                    
                    <div className="flex gap-2 p-1 bg-slate-900 rounded-xl border border-white/5">
                       {['top', 'bottom'].map(half => (
                         <button
                           key={half}
                           onClick={() => { setSelectedHalf(half); setSelectedAB(null); }}
                           className={`flex-1 py-2 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all ${
                             selectedHalf === half 
                             ? 'bg-slate-800 text-blue-400' 
                             : 'text-slate-600 hover:text-slate-400'
                           }`}
                         >
                           {half}
                         </button>
                       ))}
                    </div>
                 </div>

                 {/* Matchup List */}
                 <div className="glass-panel rounded-3xl border border-white/5 overflow-hidden">
                    <div className="p-4 bg-white/2 border-b border-white/5">
                       <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">At-Bat Sequence</p>
                    </div>
                    <div className="max-h-[500px] overflow-y-auto scrollbar-thin scrollbar-thumb-white/5">
                       {matchups.length > 0 ? matchups.map((ab, i) => (
                         <button
                           key={i}
                           onClick={() => setSelectedAB(ab)}
                           className={`w-full p-6 text-left border-b border-white/5 transition-all group flex items-center justify-between ${
                             selectedAB?.ab_index === ab.ab_index 
                             ? 'bg-blue-600/10 border-l-4 border-l-blue-500' 
                             : 'hover:bg-white/[0.02]'
                           }`}
                         >
                            <div className="space-y-1">
                               <p className="text-[8px] font-black text-blue-500 tracking-widest uppercase">AB #{ab.ab_index}</p>
                               <p className={`font-black tracking-tighter ${selectedAB?.ab_index === ab.ab_index ? 'text-white' : 'text-slate-300 group-hover:text-white'}`}>
                                 {ab.pitcher} <span className="text-slate-600 italic">vs</span> {ab.batter}
                               </p>
                            </div>
                            <ChevronRight size={16} className={selectedAB?.ab_index === ab.ab_index ? 'text-blue-500' : 'text-slate-700 opacity-0 group-hover:opacity-100 transition-all'} />
                         </button>
                       )) : (
                         <div className="p-12 text-center text-slate-600 italic text-sm">
                           No recorded entries for this inning half.
                         </div>
                       )}
                    </div>
                 </div>
              </div>

              {/* Main Center: Detail View */}
              <div className="lg:col-span-9">
                 {selectedAB ? (
                   <div className="animate-in fade-in zoom-in-95 duration-500 space-y-6">
                      
                      {/* Matchup Header Card */}
                      <div className="glass-panel p-8 rounded-[2.5rem] border border-white/5 relative overflow-hidden flex flex-col md:flex-row justify-between items-center gap-8">
                         <div className="text-center md:text-left">
                            <h2 className="text-3xl font-black italic tracking-tighter text-white mb-2 uppercase">{selectedGame.matchup}</h2>
                            <div className="flex items-center gap-3 justify-center md:justify-start">
                               <span className="px-3 py-1 bg-blue-500/10 border border-blue-500/20 rounded-lg text-blue-400 text-xs font-bold uppercase tracking-widest">
                                 Inning {selectedInning} • {selectedHalf.toUpperCase()}
                               </span>
                            </div>
                         </div>
                         <div className="flex items-center gap-6">
                            <div className="text-right">
                               <p className="text-[10px] font-black text-slate-500 uppercase">Pitcher</p>
                               <p className="font-black text-xl text-white">{selectedAB.pitcher}</p>
                            </div>
                            <div className="w-px h-10 bg-white/5"></div>
                            <div className="text-left">
                               <p className="text-[10px] font-black text-slate-500 uppercase">Batter</p>
                               <p className="font-black text-xl text-white">{selectedAB.batter}</p>
                            </div>
                         </div>
                      </div>

                      {/* Pitch Sequence Table */}
                      <div className="glass-panel rounded-[2.5rem] border border-white/5 overflow-hidden">
                         <div className="p-6 border-b border-white/5 bg-white/2">
                            <h3 className="text-sm font-black text-slate-400 uppercase tracking-widest flex items-center gap-2">
                               <Activity size={16} className="text-blue-500" /> Neural Expectation Log
                            </h3>
                         </div>
                         <div className="overflow-x-auto">
                           <table className="w-full text-left">
                             <thead>
                               <tr className="bg-slate-900/40 text-[9px] font-black text-slate-500 uppercase tracking-[0.2em] border-b border-white/5">
                                 <th className="px-8 py-4">#</th>
                                 <th className="px-8 py-4">Pitch Profile</th>
                                 <th className="px-8 py-4">Count</th>
                                 <th className="px-8 py-4 text-center">Model Prediction</th>
                                 <th className="px-8 py-4 text-right">Surprisal</th>
                               </tr>
                             </thead>
                             <tbody className="divide-y divide-white/[0.02]">
                               {selectedAB.pitches.map((p, idx) => (
                                 <tr key={idx} className="group hover:bg-white/[0.01] transition-colors">
                                   <td className="px-8 py-6 font-black text-slate-600">{idx + 1}</td>
                                   <td className="px-8 py-6">
                                      <div className="flex items-center gap-3">
                                         <div className="flex flex-col">
                                            <span className="font-bold text-white uppercase">{p.p_type}</span>
                                            <span className="text-[10px] text-slate-500">{p.fam} • {p.speed} MPH</span>
                                         </div>
                                      </div>
                                   </td>
                                   <td className="px-8 py-6">
                                      <span className="px-2 py-1 bg-slate-900 border border-white/5 rounded text-[10px] font-bold text-slate-400">{p.count}</span>
                                   </td>
                                   <td className="px-8 py-6">
                                      <div className="flex flex-col items-center gap-1">
                                         <span className={`text-[10px] font-black px-2 py-1 rounded-lg border flex items-center gap-2 ${p.fam === p.prediction ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' : 'bg-red-500/10 border-red-500/20 text-red-400'}`}>
                                            {p.prediction}
                                            {p.fam === p.prediction ? <Zap size={10} className="fill-emerald-400" /> : null}
                                         </span>
                                         <span className="text-[9px] font-bold text-slate-600 uppercase">Exp: {(p.prob * 100).toFixed(1)}%</span>
                                      </div>
                                   </td>
                                   <td className="px-8 py-6 text-right">
                                      <span className={`text-xs font-black px-3 py-1 rounded border ${p.surprisal > 6 ? 'bg-amber-500/10 border-amber-500/20 text-amber-500' : 'bg-blue-600/10 border-blue-600/20 text-blue-400'}`}>
                                         {p.surprisal.toFixed(2)}
                                      </span>
                                   </td>
                                 </tr>
                               ))}
                             </tbody>
                           </table>
                         </div>
                      </div>

                      {/* Visual Pitch Timeline */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                         <div className="glass-panel p-8 rounded-[2.5rem] border border-white/5 flex flex-col justify-center items-center text-center space-y-4">
                            <div className="bg-blue-600/20 p-4 rounded-3xl border border-blue-500/30">
                               <TrendingUp className="text-blue-400" size={32} />
                            </div>
                            <div>
                               <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Avg Surprisal</p>
                               <p className="text-4xl font-black text-white">
                                 {(selectedAB.pitches.reduce((acc, p) => acc + p.surprisal, 0) / selectedAB.pitches.length).toFixed(2)}
                               </p>
                            </div>
                         </div>
                         <div className="glass-panel p-8 rounded-[2.5rem] border border-white/5 flex flex-col justify-center items-center text-center space-y-4">
                            <div className="bg-purple-600/20 p-4 rounded-3xl border border-purple-500/30">
                               <Users className="text-purple-400" size={32} />
                            </div>
                            <div>
                               <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Prediction Accuracy</p>
                               <p className="text-4xl font-black text-white">
                                 {Math.round((selectedAB.pitches.filter(p => p.fam === p.prediction).length / selectedAB.pitches.length) * 100)}%
                               </p>
                            </div>
                         </div>
                      </div>
                   </div>
                 ) : (
                   <div className="h-full flex flex-col items-center justify-center p-20 glass-panel rounded-[2.5rem] border border-white/5 text-center space-y-6">
                      <div className="bg-slate-900 p-6 rounded-full border border-white/5">
                        <ChevronLeft size={64} className="text-slate-700 animate-[bounce_2s_infinite_horizontal]" />
                      </div>
                      <div className="space-y-2">
                        <h3 className="text-2xl font-black italic tracking-tight text-white uppercase">Initialize Analysis</h3>
                        <p className="text-slate-500 text-sm max-w-sm font-medium">Please select an At-Bat from the sequence list to begin neural deconstruction of the matchup.</p>
                      </div>
                   </div>
                 )}
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <footer className="mt-20 pt-12 border-t border-white/5 flex flex-col md:flex-row justify-between items-center gap-8 opacity-40 hover:opacity-100 transition-opacity duration-500 pb-12">
           <div className="flex items-center gap-4">
              <div className="w-10 h-10 bg-slate-900 border border-white/10 flex items-center justify-center rounded-xl">
                 <Target size={20} className="text-blue-500" />
              </div>
              <div>
                <p className="text-xs font-black text-white tracking-widest uppercase italic">MLB Pitch Bot</p>
                <p className="text-[10px] text-slate-500 font-bold">V2.4.0 • PRODUCTION GRADE</p>
              </div>
           </div>
           
           <div className="flex gap-12 text-[10px] font-bold text-slate-500 uppercase tracking-widest">
              <a href="#" className="hover:text-blue-400 transition-colors">Documentation</a>
              <a href="#" className="hover:text-blue-400 transition-colors">Privacy</a>
              <a href="#" className="hover:text-blue-400 transition-colors">Open Source</a>
           </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
