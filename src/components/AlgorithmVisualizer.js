import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, RotateCcw, ChevronRight, Settings, SkipForward, SkipBack, Zap, Brain, Code2, GitBranch, Star, Eye, Activity, AlertCircle, CheckCircle } from 'lucide-react';

const AlgorithmVisualizer = () => {
  const [code, setCode] = useState(`function bubbleSort(arr) {
  const n = arr.length;
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < n - i - 1; j++) {
      if (arr[j] > arr[j + 1]) {
        // Swap elements
        let temp = arr[j];
        arr[j] = arr[j + 1];
        arr[j + 1] = temp;
      }
    }
  }
  return arr;
}`);

  const [inputData, setInputData] = useState('[64, 34, 25, 12, 22, 11, 90]');
  const [detectedAlgorithm, setDetectedAlgorithm] = useState(null);
  const [confidence, setConfidence] = useState(0);
  const [visualData, setVisualData] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const [steps, setSteps] = useState([]);
  const [speed, setSpeed] = useState(500);
  const [visualState, setVisualState] = useState({});
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiError, setAiError] = useState(null);
  const [tokensUsed, setTokensUsed] = useState(0);
  const intervalRef = useRef(null);

  // OpenAI API Integration
  const analyzeCodeWithOpenAI = async (codeText) => {
    const apiKey = process.env.REACT_APP_OPENAI_API_KEY;
    
    if (!apiKey) {
      throw new Error('OpenAI API key not found. Please add REACT_APP_OPENAI_API_KEY to your .env file.');
    }

    const prompt = `Analyze this code and identify the algorithm. Return ONLY valid JSON in this exact format:

{
  "algorithm": "Bubble Sort",
  "type": "sorting",
  "confidence": 95,
  "timeComplexity": "O(n²)",
  "spaceComplexity": "O(1)",
  "description": "Brief explanation",
  "dataType": "array"
}

Code:
${codeText}`;

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
            {
              role: 'system',
              content: 'You are an expert algorithm analyst. Always respond with valid JSON only.'
            },
            {
              role: 'user', 
              content: prompt
            }
          ],
          temperature: 0.1,
          max_tokens: 300
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error?.message || `HTTP ${response.status}`);
      }

      const data = await response.json();
      setTokensUsed(prev => prev + (data.usage?.total_tokens || 0));
      
      let content = data.choices[0].message.content.trim();
      content = content.replace(/```json\s*/, '').replace(/```\s*$/, '');
      
      const result = JSON.parse(content);
      
      return {
        detected: true,
        ...result,
        source: 'OpenAI GPT'
      };
      
    } catch (error) {
      console.error('OpenAI API Error:', error);
      throw error;
    }
  };

  // Fallback pattern matching
  const fallbackPatternAnalysis = (codeText) => {
    const patterns = {
      bubbleSort: {
        patterns: [/for.*i.*n.*1.*for.*j.*n.*i.*1/s, /arr\[j\].*arr\[j.*1\]/],
        keywords: ['bubble', 'swap', 'nested'],
        name: 'Bubble Sort',
        type: 'sorting',
        dataType: 'array'
      },
      binarySearch: {
        patterns: [/binary/i, /mid.*left.*right/, /target/i],
        keywords: ['binary', 'mid', 'target', 'search'],
        name: 'Binary Search', 
        type: 'searching',
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
        if (normalizedCode.includes(keyword.toLowerCase())) score += 15;
      });

      if (score > highestScore && score > 20) {
        highestScore = score;
        bestMatch = {
          algorithm: pattern.name,
          type: pattern.type,
          dataType: pattern.dataType,
          confidence: Math.min((score / 100) * 100, 95),
          description: `Pattern-matched ${pattern.name} algorithm`,
          source: 'Pattern Matching (Fallback)'
        };
      }
    });

    return bestMatch;
  };

  // Smart analysis: Try AI first, fallback to patterns
  const analyzeCode = useCallback(async (codeText) => {
    if (codeText.trim().length < 50) return;
    
    setIsAnalyzing(true);
    setAiError(null);
    
    try {
      // Try OpenAI first
      const aiResult = await analyzeCodeWithOpenAI(codeText);
      setDetectedAlgorithm(aiResult);
      setConfidence(aiResult.confidence);
      
      // Auto-set input data
      if (aiResult.dataType === 'array') {
        if (aiResult.type === 'sorting') {
          setInputData('[64, 34, 25, 12, 22, 11, 90]');
        } else if (aiResult.type === 'searching') {
          setInputData('{"array": [1, 3, 5, 7, 9, 11, 13, 15], "target": 7}');
        }
      }
      
    } catch (error) {
      console.warn('AI analysis failed, using fallback:', error.message);
      setAiError(error.message);
      
      // Fallback to pattern matching
      const fallbackResult = fallbackPatternAnalysis(codeText);
      if (fallbackResult) {
        setDetectedAlgorithm(fallbackResult);
        setConfidence(fallbackResult.confidence);
        
        if (fallbackResult.dataType === 'array') {
          if (fallbackResult.type === 'sorting') {
            setInputData('[64, 34, 25, 12, 22, 11, 90]');
          } else if (fallbackResult.type === 'searching') {
            setInputData('{"array": [1, 3, 5, 7, 9, 11, 13, 15], "target": 7}');
          }
        }
      } else {
        setDetectedAlgorithm(null);
        setConfidence(0);
      }
    } finally {
      setIsAnalyzing(false);
    }
  }, []);

  // Auto-analyze code when it changes
  useEffect(() => {
    if (code.trim().length > 50) {
      const debounceTimer = setTimeout(() => {
        analyzeCode(code);
      }, 2000);

      return () => clearTimeout(debounceTimer);
    }
  }, [code, analyzeCode]);

  // Parse input data
  const parseInputData = useCallback((input, dataType) => {
    try {
      if (dataType === 'array') {
        const parsed = JSON.parse(input);
        return Array.isArray(parsed) ? parsed : parsed.array || [];
      }
      return null;
    } catch {
      return null;
    }
  }, []);

  // Generate bubble sort steps
  const generateBubbleSortSteps = (arr) => {
    const steps = [];
    const workingArray = [...arr];
    const n = workingArray.length;
    
    steps.push({
      type: 'sorting',
      array: [...workingArray],
      comparing: [],
      swapping: [],
      sorted: [],
      description: 'Initializing bubble sort algorithm'
    });

    for (let i = 0; i < n - 1; i++) {
      for (let j = 0; j < n - i - 1; j++) {
        steps.push({
          type: 'sorting',
          array: [...workingArray],
          comparing: [j, j + 1],
          swapping: [],
          sorted: [...Array(i).keys()].map(k => n - 1 - k),
          description: `Comparing: ${workingArray[j]} and ${workingArray[j + 1]}`
        });

        if (workingArray[j] > workingArray[j + 1]) {
          steps.push({
            type: 'sorting',
            array: [...workingArray],
            comparing: [j, j + 1],
            swapping: [j, j + 1],
            sorted: [...Array(i).keys()].map(k => n - 1 - k),
            description: `${workingArray[j]} > ${workingArray[j + 1]} → Swapping`
          });

          [workingArray[j], workingArray[j + 1]] = [workingArray[j + 1], workingArray[j]];

          steps.push({
            type: 'sorting',
            array: [...workingArray],
            comparing: [],
            swapping: [],
            sorted: [...Array(i).keys()].map(k => n - 1 - k),
            description: `Swapped: [${workingArray.join(', ')}]`
          });
        }
      }
    }

    steps.push({
      type: 'sorting',
      array: [...workingArray],
      comparing: [],
      swapping: [],
      sorted: [...Array(n).keys()],
      description: '✅ Sorting complete!'
    });

    return steps;
  };

  // Generate binary search steps
  const generateBinarySearchSteps = (arr, target) => {
    const steps = [];
    let left = 0;
    let right = arr.length - 1;

    steps.push({
      type: 'searching',
      array: [...arr],
      target,
      left,
      right,
      mid: null,
      found: null,
      description: `Searching for: ${target}`
    });

    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      
      steps.push({
        type: 'searching',
        array: [...arr],
        target,
        left,
        right,
        mid,
        comparing: [mid],
        description: `Checking middle: ${arr[mid]}`
      });

      if (arr[mid] === target) {
        steps.push({
          type: 'searching',
          array: [...arr],
          target,
          left,
          right,
          mid,
          found: mid,
          description: `✅ Found at index ${mid}!`
        });
        break;
      } else if (arr[mid] < target) {
        left = mid + 1;
        steps.push({
          type: 'searching',
          array: [...arr],
          target,
          left,
          right,
          mid,
          description: `${arr[mid]} < ${target} → Search right`
        });
      } else {
        right = mid - 1;
        steps.push({
          type: 'searching',
          array: [...arr],
          target,
          left,
          right,
          mid,
          description: `${arr[mid]} > ${target} → Search left`
        });
      }
    }

    return steps;
  };

  // Generate visualization steps
  const generateVisualizationSteps = (data, algorithm) => {
    if (!algorithm) return [];

    if (algorithm.algorithm?.includes('Bubble') || algorithm.type === 'sorting') {
      return generateBubbleSortSteps(data);
    } else if (algorithm.algorithm?.includes('Binary') || algorithm.type === 'searching') {
      return generateBinarySearchSteps(data.array, data.target);
    }

    return generateBubbleSortSteps(Array.isArray(data) ? data : []);
  };

  // Initialize visualization
  const runVisualization = () => {
    if (!detectedAlgorithm) {
      alert('No algorithm detected! Try writing a more complete algorithm.');
      return;
    }

    const data = parseInputData(inputData, detectedAlgorithm.dataType);
    if (!data) {
      alert('Invalid input data format!');
      return;
    }

    setVisualData(data);
    const generatedSteps = generateVisualizationSteps(data, detectedAlgorithm);
    setSteps(generatedSteps);
    setCurrentStep(0);
    setVisualState({});
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
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    }
  };

  const stepBackward = () => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  };

  // Update visual state
  useEffect(() => {
    if (steps.length > 0 && currentStep < steps.length) {
      setVisualState(steps[currentStep]);
    }
  }, [currentStep, steps]);

  // Cleanup
  useEffect(() => {
    return () => clearInterval(intervalRef.current);
  }, []);

  // Render sorting visualization
  const renderSortingVisualization = () => {
    if (!visualState.array) return null;
    
    const maxValue = Math.max(...visualState.array, 1);
    
    return (
      <div className="h-64 flex items-end justify-center gap-1 bg-gray-950 rounded-md border border-gray-800 p-4">
        {visualState.array.map((value, index) => {
          let color = 'bg-blue-600';
          let borderColor = 'border-blue-500';
          
          if (visualState.sorted?.includes(index)) {
            color = 'bg-green-600';
            borderColor = 'border-green-500';
          } else if (visualState.swapping?.includes(index)) {
            color = 'bg-red-600';
            borderColor = 'border-red-500';
          } else if (visualState.comparing?.includes(index)) {
            color = 'bg-yellow-500';
            borderColor = 'border-yellow-400';
          }
          
          return (
            <div key={index} className="flex flex-col items-center">
              <div
                className={`${color} ${borderColor} border transition-all duration-200 rounded-sm flex items-end justify-center text-white font-mono text-xs min-w-8 shadow-sm`}
                style={{
                  height: `${(value / maxValue) * 200}px`,
                  minHeight: '20px'
                }}
              >
                {value}
              </div>
              <div className="text-xs text-gray-400 mt-1 font-mono">{index}</div>
            </div>
          );
        })}
      </div>
    );
  };

  // Render searching visualization
  const renderSearchingVisualization = () => {
    if (!visualState.array) return null;
    
    return (
      <div className="h-64 bg-gray-950 rounded-md border border-gray-800 p-4">
        <div className="mb-4 text-center">
          <span className="text-sm text-gray-300">Target: </span>
          <span className="text-yellow-400 font-mono font-bold bg-gray-800 px-2 py-1 rounded text-sm">
            {visualState.target}
          </span>
        </div>
        <div className="flex items-center justify-center gap-1 h-32">
          {visualState.array.map((value, index) => {
            let color = 'bg-gray-700 border-gray-600';
            
            if (visualState.found === index) {
              color = 'bg-green-600 border-green-500';
            } else if (visualState.comparing?.includes(index)) {
              color = 'bg-yellow-500 border-yellow-400';
            } else if (visualState.left !== undefined && visualState.right !== undefined) {
              if (index >= visualState.left && index <= visualState.right) {
                color = 'bg-blue-600 border-blue-500';
              } else {
                color = 'bg-gray-800 border-gray-700';
              }
            }
            
            return (
              <div key={index} className="flex flex-col items-center">
                <div className={`${color} border transition-all duration-200 rounded w-10 h-10 flex items-center justify-center text-white font-mono text-sm shadow-sm`}>
                  {value}
                </div>
                <div className="text-xs text-gray-400 mt-1 font-mono">{index}</div>
              </div>
            );
          })}
        </div>
        {visualState.left !== undefined && visualState.right !== undefined && (
          <div className="mt-4 flex justify-center gap-6 text-sm font-mono">
            <span className="text-blue-400">Left: {visualState.left}</span>
            <span className="text-yellow-400">Mid: {visualState.mid ?? 'N/A'}</span>
            <span className="text-red-400">Right: {visualState.right}</span>
          </div>
        )}
      </div>
    );
  };

  // Main render function for visualization
  const renderVisualization = () => {
    if (!visualState.type) {
      return (
        <div className="h-64 bg-gray-950 rounded-md border border-gray-800 p-4 flex items-center justify-center text-gray-400">
          <div className="text-center">
            <Code2 className="w-16 h-16 mx-auto mb-4 text-gray-600" />
            <p className="text-gray-300 font-medium">Ready to visualize</p>
            <p className="text-sm mt-2 text-gray-500">OpenAI will analyze your code</p>
          </div>
        </div>
      );
    }

    switch (visualState.type) {
      case 'sorting':
        return renderSortingVisualization();
      case 'searching':
        return renderSearchingVisualization();
      default:
        return <div className="h-64 bg-gray-950 rounded-md border border-gray-800 p-4 flex items-center justify-center">Coming soon!</div>;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-gray-100">
      {/* GitHub-style header */}
      <header className="border-b border-gray-800 bg-gray-950">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <GitBranch className="w-6 h-6 text-gray-400" />
                <h1 className="text-xl font-semibold text-gray-100">algorithm-visualizer</h1>
              </div>
              <span className="bg-gray-800 border border-gray-700 text-gray-300 px-2 py-1 rounded-full text-xs font-medium">
                Public
              </span>
              {tokensUsed > 0 && (
                <span className="bg-blue-900 border border-blue-700 text-blue-300 px-2 py-1 rounded-full text-xs font-medium">
                  {tokensUsed} tokens
                </span>
              )}
            </div>
            <div className="flex items-center gap-3">
              <button className="flex items-center gap-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 px-3 py-1.5 rounded-md text-sm transition-colors">
                <Eye className="w-4 h-4" />
                <span>Watch</span>
                <span className="bg-gray-700 px-1.5 py-0.5 rounded text-xs ml-1">42</span>
              </button>
              <button className="flex items-center gap-1 bg-gray-800 hover:bg-gray-700 border border-gray-700 px-3 py-1.5 rounded-md text-sm transition-colors">
                <Star className="w-4 h-4" />
                <span>Star</span>
                <span className="bg-gray-700 px-1.5 py-0.5 rounded text-xs ml-1">1.2k</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto p-6">
        {/* Navigation tabs */}
        <div className="border-b border-gray-800 mb-6">
          <nav className="flex gap-6">
            <button className="flex items-center gap-2 px-3 py-2 border-b-2 border-orange-500 text-gray-100 font-medium">
              <Code2 className="w-4 h-4" />
              Code
            </button>
            <button className="flex items-center gap-2 px-3 py-2 text-gray-400 hover:text-gray-300">
              <Activity className="w-4 h-4" />
              Visualizer
            </button>
          </nav>
        </div>

        <div className="grid lg:grid-cols-2 gap-6">
          {/* Code Editor Section */}
          <div className="bg-gray-950 rounded-md border border-gray-800">
            <div className="border-b border-gray-800 px-4 py-3">
              <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                <Code2 className="w-4 h-4" />
                algorithm.js
              </h3>
            </div>
            
            {/* AI Detection Status */}
            <div className="border-b border-gray-800 px-4 py-3 bg-gray-900">
              <div className="flex items-center gap-2 mb-2">
                <Brain className={`w-4 h-4 ${isAnalyzing ? 'animate-pulse text-yellow-500' : 'text-blue-500'}`} />
                <span className="text-sm font-medium text-gray-300">
                  {isAnalyzing ? 'OpenAI Analyzing...' : 'AI Detection'}
                </span>
              </div>
              
              {aiError && (
                <div className="mb-2 flex items-center gap-2 text-red-400 text-xs">
                  <AlertCircle className="w-3 h-3" />
                  <span>AI Error: Using fallback pattern matching</span>
                </div>
              )}
              
              {detectedAlgorithm ? (
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-green-400 font-mono text-sm">{detectedAlgorithm.algorithm}</span>
                    <span className="bg-green-900 text-green-300 px-2 py-1 rounded text-xs font-mono">
                      {Math.round(confidence)}% confident
                    </span>
                  </div>
                  <p className="text-xs text-gray-400 font-mono">
                    {detectedAlgorithm.timeComplexity} time • {detectedAlgorithm.spaceComplexity} space
                  </p>
                  <p className="text-xs text-gray-500">{detectedAlgorithm.description}</p>
                  {detectedAlgorithm.source && (
                    <p className="text-xs text-gray-600">Source: {detectedAlgorithm.source}</p>
                  )}
                </div>
              ) : (
                <span className="text-gray-500 text-sm font-mono">No algorithm detected</span>
              )}
            </div>

            <div className="p-4">
              <div className="mb-4">
                <div className="bg-gray-900 rounded border border-gray-800 font-mono text-sm">
                  <div className="flex">
                    <div className="bg-gray-800 px-3 py-2 text-gray-500 text-right min-w-12 border-r border-gray-700">
                      {code.split('\n').map((_, i) => (
                        <div key={i} className="leading-6">{i + 1}</div>
                      ))}
                    </div>
                    <textarea
                      value={code}
                      onChange={(e) => setCode(e.target.value)}
                      className="flex-1 bg-transparent text-gray-100 p-2 resize-none outline-none leading-6"
                      style={{ minHeight: '300px' }}
                      placeholder="// Write your algorithm here...
// OpenAI will automatically detect and analyze it!"
                    />
                  </div>
                </div>
              </div>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Input Data
                </label>
                <input
                  type="text"
                  value={inputData}
                  onChange={(e) => setInputData(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-700 text-gray-100 p-3 rounded font-mono text-sm focus:border-blue-500 focus:outline-none"
                  placeholder="Auto-configured by AI"
                />
              </div>

              <button
                onClick={runVisualization}
                disabled={!detectedAlgorithm}
                className={`w-full py-2 px-4 rounded font-medium transition-all duration-200 flex items-center justify-center gap-2 ${
                  detectedAlgorithm
                    ? 'bg-green-600 hover:bg-green-700 text-white'
                    : 'bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700'
                }`}
              >
                <Zap className="w-4 h-4" />
                {detectedAlgorithm ? 'Run Visualization' : 'Write Algorithm First'}
              </button>
            </div>
          </div>

          {/* Visualization Section */}
          <div className="bg-gray-950 rounded-md border border-gray-800">
            <div className="border-b border-gray-800 px-4 py-3">
              <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                <Activity className="w-4 h-4" />
                Live Visualization
              </h3>
            </div>
            
            <div className="p-4">
              {/* Controls */}
              <div className="flex items-center gap-2 mb-4 flex-wrap">
                <button
                  onClick={stepBackward}
                  className="bg-gray-800 hover:bg-gray-700 border border-gray-700 p-2 rounded transition-colors"
                  disabled={steps.length === 0 || currentStep === 0}
                >
                  <SkipBack className="w-4 h-4" />
                </button>
                
                <button
                  onClick={togglePlayback}
                  className="bg-green-600 hover:bg-green-700 border border-green-500 p-2 rounded transition-colors"
                  disabled={steps.length === 0}
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </button>
                
                <button
                  onClick={stepForward}
                  className="bg-gray-800 hover:bg-gray-700 border border-gray-700 p-2 rounded transition-colors"
                  disabled={steps.length === 0 || currentStep === steps.length - 1}
                >
                  <SkipForward className="w-4 h-4" />
                </button>
                
                <button
                  onClick={resetVisualization}
                  className="bg-orange-600 hover:bg-orange-700 border border-orange-500 p-2 rounded transition-colors"
                  disabled={steps.length === 0}
                >
                  <RotateCcw className="w-4 h-4" />
                </button>

                <div className="flex items-center gap-2 ml-auto">
                  <Settings className="w-4 h-4 text-gray-400" />
                  <input
                    type="range"
                    min="100"
                    max="2000"
                    value={speed}
                    onChange={(e) => setSpeed(Number(e.target.value))}
                    className="w-20"
                  />
                  <span className="text-xs text-gray-400 font-mono">{speed}ms</span>
                </div>
              </div>

              {/* Visualization */}
              {renderVisualization()}

              {/* Step Info */}
              <div className="bg-gray-900 border border-gray-800 rounded p-4 mt-4">
                <div className="flex justify-between items-center mb-2 text-sm">
                  <span className="text-gray-400 font-mono">
                    {steps.length > 0 ? `Step ${currentStep + 1}/${steps.length}` : 'Ready'}
                  </span>
                  <span className="text-gray-400 font-mono">
                    {detectedAlgorithm ? detectedAlgorithm.algorithm : 'No algorithm'}
                  </span>
                </div>
                <p className="text-blue-300 font-mono text-sm">
                  {visualState.description || 'Write your algorithm and OpenAI will detect it automatically!'}
                </p>
              </div>
            </div>
          </div>
        </div>

        {/* Legend */}
        {visualState.type && (
          <div className="mt-6 bg-gray-950 border border-gray-800 rounded-md p-4">
            <h3 className="text-sm font-medium text-gray-300 mb-3">Legend</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-blue-600 border border-blue-500 rounded-sm"></div>
                <span className="text-gray-300 font-mono">Default</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-yellow-500 border border-yellow-400 rounded-sm"></div>
                <span className="text-gray-300 font-mono">Comparing</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-red-600 border border-red-500 rounded-sm"></div>
                <span className="text-gray-300 font-mono">Swapping</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 bg-green-600 border border-green-500 rounded-sm"></div>
                <span className="text-gray-300 font-mono">Sorted/Found</span>
              </div>
            </div>
          </div>
        )}

        {/* API Usage Info */}
        {tokensUsed > 0 && (
          <div className="mt-6 bg-blue-950 border border-blue-800 rounded-md p-4">
            <h3 className="text-sm font-medium text-blue-300 mb-2 flex items-center gap-2">
              <Brain className="w-4 h-4" />
              OpenAI Usage
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-blue-400">Tokens Used:</span>
                <span className="text-blue-200 font-mono ml-2">{tokensUsed}</span>
              </div>
              <div>
                <span className="text-blue-400">Estimated Cost:</span>
                <span className="text-blue-200 font-mono ml-2">~${((tokensUsed / 1000) * 0.002).toFixed(4)}</span>
              </div>
              <div>
                <span className="text-blue-400">Model:</span>
                <span className="text-blue-200 font-mono ml-2">GPT-3.5 Turbo</span>
              </div>
              <div>
                <span className="text-blue-400">Status:</span>
                <span className="text-green-400 font-mono ml-2 flex items-center gap-1">
                  <CheckCircle className="w-3 h-3" />
                  Active
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AlgorithmVisualizer;