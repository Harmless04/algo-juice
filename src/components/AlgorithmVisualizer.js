import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, RotateCcw, Settings, SkipForward, SkipBack, Zap, Brain, Code2, GitBranch, Star, Eye, Activity, AlertCircle, Moon, Sun } from 'lucide-react';

const AlgorithmVisualizer = () => {
  const [code, setCode] = useState('');
  const [inputData, setInputData] = useState('');
  const [detectedAlgorithm, setDetectedAlgorithm] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [steps, setSteps] = useState([]);
  const [speed, setSpeed] = useState(500);
  const [visualState, setVisualState] = useState({});
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiError, setAiError] = useState(null);
  const [tokensUsed, setTokensUsed] = useState(0);
  const [isDarkMode, setIsDarkMode] = useState(true);
  const intervalRef = useRef(null);

  // OpenAI API Integration
  const analyzeCodeWithOpenAI = async (codeText) => {
    const apiKey = process.env.REACT_APP_OPENAI_API_KEY;
    if (!apiKey) throw new Error('OpenAI API key not found');

    const prompt = `Analyze this code and identify the algorithm. Return ONLY valid JSON:
{
  "algorithm": "Algorithm Name",
  "type": "sorting|searching|graph|tree",
  "confidence": 95,
  "timeComplexity": "O(n)",
  "description": "Brief explanation",
  "dataType": "array|graph|tree"
}

Code: ${codeText}`;

    try {
      const response = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${apiKey}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: 'gpt-3.5-turbo',
          messages: [
            { role: 'system', content: 'You are an expert algorithm analyst. Respond with valid JSON only.' },
            { role: 'user', content: prompt }
          ],
          temperature: 0.1,
          max_tokens: 300
        })
      });

      const data = await response.json();
      setTokensUsed(prev => prev + (data.usage?.total_tokens || 0));
      
      let content = data.choices[0].message.content.trim();
      content = content.replace(/```json\s*/, '').replace(/```\s*$/, '');
      const result = JSON.parse(content);
      return { detected: true, ...result, source: 'OpenAI GPT' };
    } catch (error) {
      throw error;
    }
  };

  // Fallback pattern matching
  const fallbackPatternAnalysis = (codeText) => {
    const patterns = {
      bfs: {
        patterns: [/queue/i, /visited/i, /bfs/i],
        keywords: ['bfs', 'queue', 'visited', 'shift'],
        name: 'Breadth-First Search',
        type: 'graph',
        dataType: 'graph'
      },
      bubbleSort: {
        patterns: [/for.*i.*n.*for.*j.*n/s, /swap/i],
        keywords: ['bubble', 'swap', 'nested'],
        name: 'Bubble Sort',
        type: 'sorting',
        dataType: 'array'
      }
    };

    const normalizedCode = codeText.toLowerCase();
    let bestMatch = null;
    let highestScore = 0;

    Object.entries(patterns).forEach(([key, pattern]) => {
      let score = 0;
      pattern.patterns.forEach(regex => {
        if (regex.test(codeText)) score += 30;
      });
      pattern.keywords.forEach(keyword => {
        if (normalizedCode.includes(keyword)) score += 15;
      });

      if (score > highestScore && score > 20) {
        highestScore = score;
        bestMatch = {
          algorithm: pattern.name,
          type: pattern.type,
          dataType: pattern.dataType,
          confidence: Math.min(score, 95),
          description: `Pattern-matched ${pattern.name}`,
          source: 'Pattern Matching'
        };
      }
    });

    return bestMatch;
  };

  // Analyze code
  const analyzeCode = useCallback(async (codeText) => {
    if (codeText.trim().length < 50) return;
    
    setIsAnalyzing(true);
    setAiError(null);
    
    try {
      const aiResult = await analyzeCodeWithOpenAI(codeText);
      setDetectedAlgorithm(aiResult);
      setConfidence(aiResult.confidence);
      
      if (aiResult.dataType === 'graph') {
        setInputData('{"nodes": ["A", "B", "C", "D", "E"], "edges": [["A", "B"], ["A", "C"], ["B", "D"], ["C", "E"], ["D", "E"]], "startNode": "A"}');
      } else if (aiResult.dataType === 'array') {
        setInputData('[64, 34, 25, 12, 22, 11, 90]');
      }
    } catch (error) {
      setAiError(error.message);
      const fallbackResult = fallbackPatternAnalysis(codeText);
      if (fallbackResult) {
        setDetectedAlgorithm(fallbackResult);
        setConfidence(fallbackResult.confidence);
        if (fallbackResult.dataType === 'graph') {
          setInputData('{"nodes": ["A", "B", "C", "D", "E"], "edges": [["A", "B"], ["A", "C"], ["B", "D"], ["C", "E"], ["D", "E"]], "startNode": "A"}');
        } else if (fallbackResult.dataType === 'array') {
          setInputData('[64, 34, 25, 12, 22, 11, 90]');
        }
      } else {
        setDetectedAlgorithm(null);
        setConfidence(0);
      }
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  useEffect(() => {
    if (code.trim().length > 50) {
      const timer = setTimeout(() => analyzeCode(code), 2000);
      return () => clearTimeout(timer);
    }
  }, [code, analyzeCode]);

  // Parse input data
  const parseInputData = useCallback((input, dataType) => {
    try {
      const parsed = JSON.parse(input);
      if (dataType === 'graph' && parsed.nodes && parsed.edges) {
        return { nodes: parsed.nodes, edges: parsed.edges, startNode: parsed.startNode || parsed.nodes[0] };
      }
      return Array.isArray(parsed) ? parsed : parsed.array || [];
    } catch {
      return null;
    }
  }, []);

  // Generate steps
  const generateSteps = (data, algorithm) => {
    if (!algorithm) return [];
    
    if (algorithm.type === 'graph') {
      return generateBFSSteps(data);
    } else if (algorithm.type === 'sorting') {
      return generateSortSteps(data);
    }
    return [];
  };

  const generateBFSSteps = (graphData) => {
    const steps = [];
    const { nodes, edges, startNode } = graphData;
    const graph = {};
    nodes.forEach(node => graph[node] = []);
    edges.forEach(([from, to]) => {
      graph[from].push(to);
      graph[to].push(from);
    });

    const visited = new Set();
    const queue = [startNode];
    const result = [];

    steps.push({
      type: 'graph',
      graph: graphData,
      queue: [...queue],
      visited: new Set(),
      current: null,
      result: [],
      description: `Starting BFS from ${startNode}`
    });

    while (queue.length > 0) {
      const node = queue.shift();
      if (!visited.has(node)) {
        visited.add(node);
        result.push(node);
        
        steps.push({
          type: 'graph',
          graph: graphData,
          queue: [...queue],
          visited: new Set(visited),
          current: node,
          result: [...result],
          description: `Visiting ${node}`
        });

        for (let neighbor of graph[node] || []) {
          if (!visited.has(neighbor) && !queue.includes(neighbor)) {
            queue.push(neighbor);
          }
        }
      }
    }

    return steps;
  };

  const generateSortSteps = (arr) => {
    const steps = [];
    const workingArray = [...arr];
    const n = workingArray.length;
    
    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < n - i - 1; j++) {
        steps.push({
          type: 'sorting',
          array: [...workingArray],
          comparing: [j, j + 1],
          description: `Comparing ${workingArray[j]} and ${workingArray[j + 1]}`
        });

        if (workingArray[j] > workingArray[j + 1]) {
          [workingArray[j], workingArray[j + 1]] = [workingArray[j + 1], workingArray[j]];
          steps.push({
            type: 'sorting',
            array: [...workingArray],
            swapping: [j, j + 1],
            description: 'Swapped!'
          });
        }
      }
    }
    return steps;
  };

  // Run visualization
  const runVisualization = () => {
    if (!detectedAlgorithm) {
      alert('No algorithm detected!');
      return;
    }

    const data = parseInputData(inputData, detectedAlgorithm.dataType);
    if (!data) {
      alert('Invalid input data!');
      return;
    }

    const generatedSteps = generateSteps(data, detectedAlgorithm);
    setSteps(generatedSteps);
    setCurrentStep(0);
    setIsPlaying(false);
  };

  // Playback controls
  const togglePlayback = () => {
    if (isPlaying) {
      clearInterval(intervalRef.current);
      setIsPlaying(false);
    } else {
      setIsPlaying(true);
      intervalRef.current = setInterval(() => {
        setCurrentStep(prev => {
          if (prev >= steps.length - 1) {
            setIsPlaying(false);
            clearInterval(intervalRef.current);
            return prev;
          }
          return prev + 1;
        });
      }, speed);
    }
  };

  const resetVisualization = () => {
    clearInterval(intervalRef.current);
    setIsPlaying(false);
    setCurrentStep(0);
  };

  const stepForward = () => {
    if (currentStep < steps.length - 1) setCurrentStep(currentStep + 1);
  };

  const stepBackward = () => {
    if (currentStep > 0) setCurrentStep(currentStep - 1);
  };

  useEffect(() => {
    if (steps.length > 0 && currentStep < steps.length) {
      setVisualState(steps[currentStep]);
    }
  }, [currentStep, steps]);

  useEffect(() => {
    return () => clearInterval(intervalRef.current);
  }, []);

  // Theme styles
  const theme = {
    bg: isDarkMode ? 'bg-gray-900' : 'bg-gray-50',
    card: isDarkMode ? 'bg-gray-950' : 'bg-white',
    text: isDarkMode ? 'text-gray-100' : 'text-gray-900',
    textMuted: isDarkMode ? 'text-gray-400' : 'text-gray-500',
    border: isDarkMode ? 'border-gray-800' : 'border-gray-200',
    editor: isDarkMode ? 'bg-gray-900' : 'bg-gray-100',
    button: isDarkMode ? 'bg-gray-800 hover:bg-gray-700' : 'bg-gray-200 hover:bg-gray-300'
  };

  // Render visualizations
  const renderVisualization = () => {
    if (!visualState.type) {
      return (
        <div className={`h-64 ${theme.card} rounded-md border ${theme.border} p-4 flex items-center justify-center`}>
          <div className="text-center">
            <Code2 className={`w-16 h-16 mx-auto mb-4 ${theme.textMuted}`} />
            <p className={theme.text}>Ready to visualize</p>
            <p className={`text-sm mt-2 ${theme.textMuted}`}>AI will analyze your code</p>
          </div>
        </div>
      );
    }

    if (visualState.type === 'graph') {
      return renderGraph();
    } else if (visualState.type === 'sorting') {
      return renderSorting();
    }

    return <div className={`h-64 ${theme.card} rounded-md border ${theme.border} p-4 flex items-center justify-center`}>Coming soon!</div>;
  };

  const renderGraph = () => {
    const { nodes, edges } = visualState.graph;
    const nodePositions = {};
    nodes.forEach((node, i) => {
      const angle = (i / nodes.length) * 2 * Math.PI;
      nodePositions[node] = {
        x: 150 + 100 * Math.cos(angle),
        y: 120 + 100 * Math.sin(angle)
      };
    });

    return (
      <div className={`h-64 ${theme.card} rounded-md border ${theme.border} p-4`}>
        <div className="mb-4 text-center">
          <span className="text-blue-500 font-mono">
            {visualState.current ? `Current: ${visualState.current}` : 'Graph Traversal'}
          </span>
        </div>
        
        <div className="flex justify-center">
          <svg width="300" height="180" viewBox="0 0 300 180">
            {edges.map(([from, to], i) => (
              <line key={i} x1={nodePositions[from]?.x} y1={nodePositions[from]?.y}
                    x2={nodePositions[to]?.x} y2={nodePositions[to]?.y}
                    stroke={isDarkMode ? "#6b7280" : "#9ca3af"} strokeWidth="2" />
            ))}
            
            {nodes.map(node => {
              const isVisited = visualState.visited?.has(node);
              const isCurrent = visualState.current === node;
              const inQueue = visualState.queue?.includes(node);
              
              let nodeColor = isDarkMode ? '#374151' : '#d1d5db';
              let textColor = isDarkMode ? '#d1d5db' : '#374151';
              
              if (isCurrent) {
                nodeColor = '#eab308';
                textColor = '#111827';
              } else if (isVisited) {
                nodeColor = '#16a34a';
                textColor = '#ffffff';
              } else if (inQueue) {
                nodeColor = '#3b82f6';
                textColor = '#ffffff';
              }
              
              return (
                <g key={node}>
                  <circle cx={nodePositions[node]?.x} cy={nodePositions[node]?.y} r="20"
                          fill={nodeColor} stroke={isDarkMode ? "#4b5563" : "#9ca3af"} strokeWidth="2" />
                  <text x={nodePositions[node]?.x} y={nodePositions[node]?.y + 5}
                        textAnchor="middle" fill={textColor} fontSize="14" fontWeight="bold">
                    {node}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
        
        <div className="mt-4 text-sm text-center">
          <span className="text-blue-500">Queue: </span>
          <span className="text-blue-400 font-mono">[{visualState.queue?.join(', ') || ''}]</span>
          <span className="ml-4 text-green-500">Visited: </span>
          <span className="text-green-400 font-mono">[{visualState.result?.join(', ') || ''}]</span>
        </div>
      </div>
    );
  };

  const renderSorting = () => {
    const maxValue = Math.max(...visualState.array, 1);
    
    return (
      <div className={`h-64 flex items-end justify-center gap-1 ${theme.card} rounded-md border ${theme.border} p-4`}>
        {visualState.array.map((value, index) => {
          let color = isDarkMode ? 'bg-blue-600' : 'bg-blue-500';
          if (visualState.swapping?.includes(index)) {
            color = isDarkMode ? 'bg-red-600' : 'bg-red-500';
          } else if (visualState.comparing?.includes(index)) {
            color = isDarkMode ? 'bg-yellow-500' : 'bg-yellow-400';
          }
          
          return (
            <div key={index} className="flex flex-col items-center">
              <div className={`${color} transition-all duration-200 rounded-sm flex items-end justify-center text-white font-mono text-xs min-w-8`}
                   style={{ height: `${(value / maxValue) * 200}px`, minHeight: '20px' }}>
                {value}
              </div>
              <div className={`text-xs ${theme.textMuted} mt-1`}>{index}</div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div className={`min-h-screen ${theme.bg} ${theme.text}`}>
      {/* Header */}
      <header className={`border-b ${theme.border} ${theme.card}`}>
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <GitBranch className={`w-6 h-6 ${theme.textMuted}`} />
                <h1 className="text-xl font-semibold">algorithm-visualizer</h1>
              </div>
              <span className={`${theme.button} border ${theme.border} px-2 py-1 rounded-full text-xs`}>Public</span>
              {tokensUsed > 0 && (
                <span className="bg-blue-900 text-blue-300 px-2 py-1 rounded-full text-xs">{tokensUsed} tokens</span>
              )}
            </div>
            <div className="flex items-center gap-3">
              <button onClick={() => setIsDarkMode(!isDarkMode)}
                      className={`${theme.button} border ${theme.border} p-2 rounded-md transition-colors`}>
                {isDarkMode ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
              </button>
              <button className={`flex items-center gap-1 ${theme.button} border ${theme.border} px-3 py-1.5 rounded-md text-sm`}>
                <Eye className="w-4 h-4" />
                <span>Watch</span>
                <span className={`${theme.button} px-1.5 py-0.5 rounded text-xs ml-1`}>42</span>
              </button>
              <button className={`flex items-center gap-1 ${theme.button} border ${theme.border} px-3 py-1.5 rounded-md text-sm`}>
                <Star className="w-4 h-4" />
                <span>Star</span>
                <span className={`${theme.button} px-1.5 py-0.5 rounded text-xs ml-1`}>1.2k</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto p-6">
        {/* Navigation */}
        <div className={`border-b ${theme.border} mb-6`}>
          <nav className="flex gap-6">
            <button className={`flex items-center gap-2 px-3 py-2 border-b-2 border-orange-500 ${theme.text} font-medium`}>
              <Code2 className="w-4 h-4" />
              Code
            </button>
            <button className={`flex items-center gap-2 px-3 py-2 ${theme.textMuted}`}>
              <Activity className="w-4 h-4" />
              Visualizer
            </button>
          </nav>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Code Editor */}
          <div className={`${theme.card} rounded-md border ${theme.border}`}>
            <div className={`border-b ${theme.border} px-4 py-3`}>
              <h3 className={`text-sm font-medium ${theme.textMuted} flex items-center gap-2`}>
                <Code2 className="w-4 h-4" />
                algorithm.js
              </h3>
            </div>
            
            {/* AI Detection */}
            <div className={`border-b ${theme.border} px-4 py-3 ${theme.editor}`}>
              <div className="flex items-center gap-2 mb-2">
                <Brain className={`w-4 h-4 ${isAnalyzing ? 'animate-pulse text-yellow-500' : 'text-blue-500'}`} />
                <span className={`text-sm font-medium ${theme.textMuted}`}>
                  {isAnalyzing ? 'AI Analyzing...' : 'AI Detection'}
                </span>
              </div>
              
              {aiError && (
                <div className="mb-2 flex items-center gap-2 text-red-400 text-xs">
                  <AlertCircle className="w-3 h-3" />
                  <span>Using fallback detection</span>
                </div>
              )}
              
              {detectedAlgorithm ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-green-400 font-mono text-sm">{detectedAlgorithm.algorithm}</span>
                    <span className="bg-green-900 text-green-300 px-2 py-1 rounded text-xs">
                      {Math.round(confidence)}% confident
                    </span>
                  </div>
                  <p className={`text-xs ${theme.textMuted}`}>{detectedAlgorithm.description}</p>
                  <p className={`text-xs ${theme.textMuted}`}>Source: {detectedAlgorithm.source}</p>
                </div>
              ) : (
                <span className={`${theme.textMuted} text-sm`}>No algorithm detected</span>
              )}
            </div>

            <div className="p-4">
              <div className="mb-4">
                <div className={`${theme.editor} rounded border ${theme.border} font-mono text-sm`}>
                  <div className="flex">
                    <div className={`${isDarkMode ? 'bg-gray-800' : 'bg-gray-200'} px-3 py-2 ${theme.textMuted} text-right min-w-12 border-r ${theme.border}`}>
                      {code.split('\n').map((_, i) => (
                        <div key={i} className="leading-6">{i + 1}</div>
                      ))}
                    </div>
                    <textarea
                      value={code}
                      onChange={(e) => setCode(e.target.value)}
                      className={`flex-1 bg-transparent ${theme.text} p-2 resize-none outline-none leading-6`}
                      style={{ minHeight: '300px' }}
                      placeholder="// Write your algorithm here...
// Examples: BFS, DFS, bubble sort, binary search
// AI will automatically detect and visualize it!"
                    />
                  </div>
                </div>
              </div>
              
              <div className="mb-4">
                <label className={`block text-sm font-medium ${theme.textMuted} mb-2`}>Input Data</label>
                <input
                  type="text"
                  value={inputData}
                  onChange={(e) => setInputData(e.target.value)}
                  className={`w-full ${theme.editor} border ${theme.border} ${theme.text} p-3 rounded font-mono text-sm focus:border-blue-500 focus:outline-none`}
                  placeholder="Input data will be auto-configured"
                />
              </div>

              <button
                onClick={runVisualization}
                disabled={!detectedAlgorithm}
                className={`w-full py-2 px-4 rounded font-medium flex items-center justify-center gap-2 transition-colors ${
                  detectedAlgorithm
                    ? 'bg-green-600 hover:bg-green-700 text-white'
                    : `${theme.button} ${theme.textMuted} cursor-not-allowed`
                }`}
              >
                <Zap className="w-4 h-4" />
                {detectedAlgorithm ? 'Run Visualization' : 'Write Algorithm First'}
              </button>
            </div>
          </div>

          {/* Visualization */}
          <div className={`${theme.card} rounded-md border ${theme.border}`}>
            <div className={`border-b ${theme.border} px-4 py-3`}>
              <h3 className={`text-sm font-medium ${theme.textMuted} flex items-center gap-2`}>
                <Activity className="w-4 h-4" />
                Live Visualization
              </h3>
            </div>
            
            <div className="p-4">
              {/* Controls */}
              <div className="flex items-center gap-2 mb-4">
                <button onClick={stepBackward} className={`${theme.button} p-2 rounded`}
                        disabled={steps.length === 0 || currentStep === 0}>
                  <SkipBack className="w-4 h-4" />
                </button>
                
                <button onClick={togglePlayback} className="bg-green-600 hover:bg-green-700 p-2 rounded text-white"
                        disabled={steps.length === 0}>
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
                
                <button onClick={stepForward} className={`${theme.button} p-2 rounded`}
                        disabled={steps.length === 0 || currentStep === steps.length - 1}>
                  <SkipForward className="w-4 h-4" />
                </button>
                
                <button onClick={resetVisualization} className="bg-orange-600 hover:bg-orange-700 p-2 rounded text-white"
                        disabled={steps.length === 0}>
                  <RotateCcw className="w-4 h-4" />
                </button>

                <div className="flex items-center gap-2 ml-auto">
                  <Settings className={`w-4 h-4 ${theme.textMuted}`} />
                  <input type="range" min="100" max="2000" value={speed}
                         onChange={(e) => setSpeed(Number(e.target.value))} className="w-20" />
                  <span className={`text-xs ${theme.textMuted}`}>{speed}ms</span>
                </div>
              </div>

              {/* Visualization */}
              {renderVisualization()}

              {/* Step Info */}
              <div className={`${theme.editor} border ${theme.border} rounded p-4 mt-4`}>
                <div className="flex justify-between items-center mb-2 text-sm">
                  <span className={`${theme.textMuted} font-mono`}>
                    {steps.length > 0 ? `Step ${currentStep + 1}/${steps.length}` : 'Ready'}
                  </span>
                  <span className={`${theme.textMuted} font-mono`}>
                    {detectedAlgorithm ? detectedAlgorithm.algorithm : 'No algorithm'}
                  </span>
                </div>
                <p className="text-blue-500 font-mono text-sm">
                  {visualState.description || 'Write your algorithm and AI will detect it automatically!'}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AlgorithmVisualizer;