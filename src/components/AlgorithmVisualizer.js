import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Play, Pause, RotateCcw, Settings, SkipForward, SkipBack, Zap, Brain, Code2, GitBranch, Star, Eye, Activity, AlertCircle, Moon, Sun } from 'lucide-react';

const generateBFSSteps = (graphData) => {
    const steps = [];
    
    // Validate input data
    if (!graphData || !graphData.nodes || !graphData.edges) {
      console.error('Invalid graph data:', graphData);
      return [{
        type: 'graph',
        graph: { nodes: ['A'], edges: [], startNode: 'A' },
        queue: [],
        visited: new Set(),
        current: null,
        result: [],
        description: 'Invalid graph data provided'
      }];
    }

    const { nodes, edges, startNode } = graphData;
    const graph = {};
    
    // Initialize graph adjacency list
    nodes.forEach(node => graph[node] = []);
    
    // Build adjacency list safely
    edges.forEach(edge => {
      if (Array.isArray(edge) && edge.length >= 2) {
        const [from, to] = edge;
        if (graph[from] && graph[to]) {
          graph[from].push(to);
          graph[to].push(from);
        }
      }
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

        // Add neighbors safely
        const neighbors = graph[node] || [];
        for (let neighbor of neighbors) {
          if (!visited.has(neighbor) && !queue.includes(neighbor)) {
            queue.push(neighbor);
          }
        }
      }
    }

    steps.push({
      type: 'graph',
      graph: graphData,
      queue: [],
      visited,
      current: null,
      result,
      description: `BFS complete! Order: [${result.join(', ')}]`
    });

    return steps;
  };

  const generateDFSSteps = (graphData) => {
    const steps = [];
    const { nodes, edges, startNode } = graphData;
    const graph = {};
    nodes.forEach(node => graph[node] = []);
    edges.forEach(edge => {
      const [from, to] = edge;
      graph[from].push(to);
      graph[to].push(from);
    });

    const visited = new Set();
    const stack = [startNode];
    const result = [];

    steps.push({
      type: 'graph',
      graph: graphData,
      stack: [...stack],
      visited: new Set(),
      current: null,
      result: [],
      description: `Starting DFS from ${startNode}`
    });

    while (stack.length > 0) {
      const node = stack.pop();
      if (!visited.has(node)) {
        visited.add(node);
        result.push(node);
        
        steps.push({
          type: 'graph',
          graph: graphData,
          stack: [...stack],
          visited: new Set(visited),
          current: node,
          result: [...result],
          description: `Visiting ${node}`
        });

        for (let neighbor of graph[node] || []) {
          if (!visited.has(neighbor) && !stack.includes(neighbor)) {
            stack.push(neighbor);
          }
        }
      }
    }

    return steps;
  };

  const generateDijkstraSteps = (graphData) => {
    const steps = [];
    const { nodes, edges, startNode } = graphData;
    const graph = {};
    nodes.forEach(node => graph[node] = []);
    edges.forEach(([from, to, weight]) => {
      graph[from].push({ node: to, weight });
      graph[to].push({ node: from, weight });
    });

    const distances = {};
    const visited = new Set();
    const previous = {};
    
    // Initialize distances
    nodes.forEach(node => {
      distances[node] = node === startNode ? 0 : Infinity;
      previous[node] = null;
    });

    steps.push({
      type: 'weighted_graph',
      graph: graphData,
      distances: {...distances},
      visited: new Set(),
      current: null,
      description: `Initializing Dijkstra from ${startNode}`
    });

    while (visited.size < nodes.length) {
      // Find unvisited node with minimum distance
      let current = null;
      let minDistance = Infinity;
      
      for (let node of nodes) {
        if (!visited.has(node) && distances[node] < minDistance) {
          minDistance = distances[node];
          current = node;
        }
      }

      if (current === null) break;

      visited.add(current);
      
      steps.push({
        type: 'weighted_graph',
        graph: graphData,
        distances: {...distances},
        visited: new Set(visited),
        current,
        description: `Processing node ${current} with distance ${distances[current]}`
      });

      // Update distances to neighbors
      for (let neighbor of graph[current] || []) {
        if (!visited.has(neighbor.node)) {
          const newDistance = distances[current] + neighbor.weight;
          if (newDistance < distances[neighbor.node]) {
            distances[neighbor.node] = newDistance;
            previous[neighbor.node] = current;
            
            steps.push({
              type: 'weighted_graph',
              graph: graphData,
              distances: {...distances},
              visited: new Set(visited),
              current,
              updatedNode: neighbor.node,
              description: `Updated distance to ${neighbor.node}: ${newDistance}`
            });
          }
        }
      }
    }

    return steps;
  };

  const generateFibonacciSteps = (n) => {
    const steps = [];
    
    // Validate input
    if (typeof n !== 'number' || n < 0) {
      return [{
        type: 'dp',
        dp: [0],
        current: 0,
        description: 'Invalid input for Fibonacci'
      }];
    }

    const dp = new Array(n + 1).fill(0);
    
    if (n === 0) {
      return [{
        type: 'dp',
        dp: [0],
        current: 0,
        description: 'F(0) = 0'
      }];
    }
    
    if (n === 1) {
      steps.push({
        type: 'dp',
        dp: [0],
        current: 0,
        description: 'Base case: F(0) = 0'
      });
      
      steps.push({
        type: 'dp',
        dp: [0, 1],
        current: 1,
        description: 'Base case: F(1) = 1'
      });
      
      return steps;
    }

    // Base cases
    dp[0] = 0;
    dp[1] = 1;
    
    steps.push({
      type: 'dp',
      dp: [0],
      current: 0,
      description: 'Base case: F(0) = 0'
    });
    
    steps.push({
      type: 'dp',
      dp: [0, 1],
      current: 1,
      description: 'Base case: F(1) = 1'
    });

    // Build up the sequence
    for (let i = 2; i <= n; i++) {
      dp[i] = dp[i-1] + dp[i-2];
      steps.push({
        type: 'dp',
        dp: [...dp.slice(0, i+1)],
        current: i,
        description: `F(${i}) = F(${i-1}) + F(${i-2}) = ${dp[i-1]} + ${dp[i-2]} = ${dp[i]}`
      });
    }

    steps.push({
      type: 'dp',
      dp: [...dp],
      current: n,
      description: `Fibonacci sequence complete! F(${n}) = ${dp[n]}`
    });

    return steps;
  };

  const generateKnapsackSteps = (data) => {
    const { capacity, weights, values } = data;
    const n = weights.length;
    const dp = Array(n + 1).fill(null).map(() => Array(capacity + 1).fill(0));
    const steps = [];

    steps.push({
      type: 'dp_table',
      dp: dp.map(row => [...row]),
      current: { i: 0, w: 0 },
      items: weights.map((w, i) => ({ weight: w, value: values[i], index: i })),
      description: 'Initializing 0/1 Knapsack DP table'
    });

    for (let i = 1; i <= n; i++) {
      for (let w = 0; w <= capacity; w++) {
        if (weights[i-1] <= w) {
          const include = values[i-1] + dp[i-1][w - weights[i-1]];
          const exclude = dp[i-1][w];
          dp[i][w] = Math.max(include, exclude);
          
          steps.push({
            type: 'dp_table',
            dp: dp.map(row => [...row]),
            current: { i, w },
            comparing: { include, exclude },
            description: `Item ${i}: weight=${weights[i-1]}, value=${values[i-1]}. Max(include: ${include}, exclude: ${exclude}) = ${dp[i][w]}`
          });
        } else {
          dp[i][w] = dp[i-1][w];
          steps.push({
            type: 'dp_table',
            dp: dp.map(row => [...row]),
            current: { i, w },
            description: `Item ${i} too heavy for capacity ${w}, taking previous best: ${dp[i][w]}`
          });
        }
      }
    }

    return steps;
  };

  const generateLCSSteps = (data) => {
    const { str1, str2 } = data;
    const m = str1.length;
    const n = str2.length;
    const dp = Array(m + 1).fill(null).map(() => Array(n + 1).fill(0));
    const steps = [];

    steps.push({
      type: 'lcs_table',
      dp: dp.map(row => [...row]),
      str1,
      str2,
      current: { i: 0, j: 0 },
      description: 'Initializing LCS table'
    });

    for (let i = 1; i <= m; i++) {
      for (let j = 1; j <= n; j++) {
        if (str1[i-1] === str2[j-1]) {
          dp[i][j] = dp[i-1][j-1] + 1;
          steps.push({
            type: 'lcs_table',
            dp: dp.map(row => [...row]),
            str1,
            str2,
            current: { i, j },
            match: true,
            description: `Characters match: '${str1[i-1]}' = '${str2[j-1]}', LCS length = ${dp[i][j]}`
          });
        } else {
          dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
          steps.push({
            type: 'lcs_table',
            dp: dp.map(row => [...row]),
            str1,
            str2,
            current: { i, j },
            match: false,
            description: `Characters don't match: '${str1[i-1]}' ≠ '${str2[j-1]}', taking max(${dp[i-1][j]}, ${dp[i][j-1]}) = ${dp[i][j]}`
          });
        }
      }
    }

    return steps;
  };

  const generateQuickSortSteps = (arr) => {
    const steps = [];
    const workingArray = [...arr];
    
    const quickSort = (arr, low, high) => {
      if (low < high) {
        const pi = partition(arr, low, high);
        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
      }
    };

    const partition = (arr, low, high) => {
      const pivot = arr[high];
      let i = low - 1;

      steps.push({
        type: 'sorting',
        array: [...arr],
        pivot: high,
        partitioning: [low, high],
        description: `Partitioning with pivot ${pivot}`
      });

      for (let j = low; j < high; j++) {
        if (arr[j] < pivot) {
          i++;
          [arr[i], arr[j]] = [arr[j], arr[i]];
          steps.push({
            type: 'sorting',
            array: [...arr],
            pivot: high,
            swapping: [i, j],
            description: `Swapping ${arr[j]} and ${arr[i]}`
          });
        }
      }
      
      [arr[i + 1], arr[high]] = [arr[high], arr[i + 1]];
      steps.push({
        type: 'sorting',
        array: [...arr],
        pivot: i + 1,
        description: `Pivot ${pivot} in correct position`
      });
      
      return i + 1;
    };

    quickSort(workingArray, 0, workingArray.length - 1);
    return steps;
  };

  const generateMergeSortSteps = (arr) => {
    const steps = [];
    const workingArray = [...arr];
    
    const mergeSort = (arr, left, right) => {
      if (left < right) {
        const mid = Math.floor((left + right) / 2);
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
      }
    };

    const merge = (arr, left, mid, right) => {
      const leftArr = arr.slice(left, mid + 1);
      const rightArr = arr.slice(mid + 1, right + 1);
      
      steps.push({
        type: 'sorting',
        array: [...arr],
        merging: [left, mid, right],
        leftArray: leftArr,
        rightArray: rightArr,
        description: `Merging subarrays [${left}...${mid}] and [${mid+1}...${right}]`
      });

      let i = 0, j = 0, k = left;
      
      while (i < leftArr.length && j < rightArr.length) {
        if (leftArr[i] <= rightArr[j]) {
          arr[k] = leftArr[i];
          i++;
        } else {
          arr[k] = rightArr[j];
          j++;
        }
        
        steps.push({
          type: 'sorting',
          array: [...arr],
          merging: [left, mid, right],
          current: k,
          description: `Placed ${arr[k]} at position ${k}`
        });
        k++;
      }

      while (i < leftArr.length) {
        arr[k] = leftArr[i];
        steps.push({
          type: 'sorting',
          array: [...arr],
          current: k,
          description: `Copying remaining element ${arr[k]}`
        });
        i++;
        k++;
      }

      while (j < rightArr.length) {
        arr[k] = rightArr[j];
        steps.push({
          type: 'sorting',
          array: [...arr],
          current: k,
          description: `Copying remaining element ${arr[k]}`
        });
        j++;
        k++;
      }
    };

    mergeSort(workingArray, 0, workingArray.length - 1);
    return steps;
  };

  const generateBinarySearchSteps = (data) => {
    const { array, target } = data;
    const steps = [];
    let left = 0;
    let right = array.length - 1;

    while (left <= right) {
      const mid = Math.floor((left + right) / 2);
      
      steps.push({
        type: 'searching',
        array,
        target,
        left,
        right,
        mid,
        comparing: [mid],
        description: `Checking middle element: ${array[mid]}`
      });

      if (array[mid] === target) {
        steps.push({
          type: 'searching',
          array,
          target,
          found: mid,
          description: `Found ${target} at index ${mid}!`
        });
        break;
      } else if (array[mid] < target) {
        left = mid + 1;
        steps.push({
          type: 'searching',
          array,
          target,
          left,
          right,
          description: `${array[mid]} < ${target}, search right half`
        });
      } else {
        right = mid - 1;
        steps.push({
          type: 'searching',
          array,
          target,
          left,
          right,
          description: `${array[mid]} > ${target}, search left half`
        });
      }
    }

    return steps;
  };

  const generateTreeTraversalSteps = (tree) => {
    const steps = [];
    const result = [];
    
    const traverse = (node) => {
      if (!node) return;
      
      // Inorder: left, root, right
      steps.push({
        type: 'tree',
        tree,
        visiting: node.value,
        result: [...result],
        description: `Visiting node ${node.value}`
      });
      
      if (node.left) traverse(node.left);
      
      result.push(node.value);
      steps.push({
        type: 'tree',
        tree,
        processing: node.value,
        result: [...result],
        description: `Processing node ${node.value}`
      });
      
      if (node.right) traverse(node.right);
    };
    
    traverse(tree);
    return steps;
  };

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
  
  // Real-time features
  const [watchCount, setWatchCount] = useState(42);
  const [starCount, setStarCount] = useState(1247);
  const [isWatching, setIsWatching] = useState(false);
  const [isStarred, setIsStarred] = useState(false);
  const [viewersCount, setViewersCount] = useState(8);
  
  const intervalRef = useRef(null);
  const realtimeRef = useRef(null);
  const codeInputRef = useRef(null);

  // Real-time data simulation
  useEffect(() => {
    // Simulate real-time viewer activity
    realtimeRef.current = setInterval(() => {
      // Randomly fluctuate viewer count (simulates people coming and going)
      setViewersCount(prev => {
        const change = Math.floor(Math.random() * 3) - 1; // -1, 0, or 1
        const newCount = prev + change;
        return Math.max(1, Math.min(newCount, 25)); // Keep between 1-25 viewers
      });

      // Occasionally add stars (much less frequent)
      if (Math.random() < 0.02) { // 2% chance every 5 seconds = roughly 1 star per 4 minutes
        setStarCount(prev => prev + 1);
      }

      // Occasionally add watchers (less frequent than viewers)
      if (Math.random() < 0.01) { // 1% chance every 5 seconds
        setWatchCount(prev => prev + 1);
      }
    }, 5000); // Update every 5 seconds

    return () => clearInterval(realtimeRef.current);
  }, []);

  // Handle watch button
  const handleWatch = () => {
    setIsWatching(!isWatching);
    setWatchCount(prev => isWatching ? prev - 1 : prev + 1);
  };

  // Handle star button
  const handleStar = () => {
    setIsStarred(!isStarred);
    setStarCount(prev => isStarred ? prev - 1 : prev + 1);
  };

  // Format large numbers (e.g., 1247 -> 1.2k)
  const formatCount = (count) => {
    if (count >= 1000) {
      return (count / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
    }
    return count.toString();
  };

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
      dfs: {
        patterns: [/stack/i, /visited/i, /dfs/i, /recursive/i],
        keywords: ['dfs', 'stack', 'visited', 'recursive', 'depth'],
        name: 'Depth-First Search',
        type: 'graph',
        dataType: 'graph'
      },
      bubbleSort: {
        patterns: [/for.*i.*n.*for.*j.*n/s, /swap/i],
        keywords: ['bubble', 'swap', 'nested'],
        name: 'Bubble Sort',
        type: 'sorting',
        dataType: 'array'
      },
      quickSort: {
        patterns: [/pivot/i, /partition/i, /quickSort/i],
        keywords: ['pivot', 'partition', 'quicksort', 'recursive'],
        name: 'Quick Sort',
        type: 'sorting',
        dataType: 'array'
      },
      mergeSort: {
        patterns: [/merge/i, /mergeSort/i, /divide/i],
        keywords: ['merge', 'mergesort', 'divide', 'conquer'],
        name: 'Merge Sort',
        type: 'sorting',
        dataType: 'array'
      },
      binarySearch: {
        patterns: [/binary/i, /mid/i, /left.*right/i],
        keywords: ['binary', 'search', 'mid', 'left', 'right'],
        name: 'Binary Search',
        type: 'searching',
        dataType: 'array'
      },
      fibonacci: {
        patterns: [/fibonacci/i, /fib/i, /dp/i, /memo/i],
        keywords: ['fibonacci', 'fib', 'dp', 'memo', 'dynamic'],
        name: 'Fibonacci (Dynamic Programming)',
        type: 'dynamic_programming',
        dataType: 'dp'
      },
      knapsack: {
        patterns: [/knapsack/i, /dp.*weight/i, /value.*weight/i],
        keywords: ['knapsack', 'weight', 'value', 'dp', 'dynamic'],
        name: '0/1 Knapsack',
        type: 'dynamic_programming',
        dataType: 'dp'
      },
      lcs: {
        patterns: [/lcs/i, /longest.*common/i, /subsequence/i],
        keywords: ['lcs', 'longest', 'common', 'subsequence', 'dp'],
        name: 'Longest Common Subsequence',
        type: 'dynamic_programming',
        dataType: 'dp'
      },
      dijkstra: {
        patterns: [/dijkstra/i, /shortest.*path/i, /priority.*queue/i],
        keywords: ['dijkstra', 'shortest', 'path', 'priority', 'distance'],
        name: "Dijkstra's Algorithm",
        type: 'graph',
        dataType: 'weighted_graph'
      },
      kruskal: {
        patterns: [/kruskal/i, /mst/i, /minimum.*spanning/i],
        keywords: ['kruskal', 'mst', 'spanning', 'tree', 'union', 'find'],
        name: "Kruskal's MST",
        type: 'graph',
        dataType: 'weighted_graph'
      },
      treeTraversal: {
        patterns: [/inorder|preorder|postorder/i, /left.*right/i, /traverse/i],
        keywords: ['inorder', 'preorder', 'postorder', 'traverse', 'tree'],
        name: 'Tree Traversal',
        type: 'tree',
        dataType: 'tree'
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
      } else if (aiResult.dataType === 'weighted_graph') {
        setInputData('{"nodes": ["A", "B", "C", "D", "E"], "edges": [["A", "B", 4], ["A", "C", 2], ["B", "D", 3], ["C", "E", 1], ["D", "E", 5]], "startNode": "A"}');
      } else if (aiResult.dataType === 'array') {
        setInputData('[64, 34, 25, 12, 22, 11, 90]');
      } else if (aiResult.dataType === 'dp') {
        if (aiResult.algorithm.includes('Fibonacci')) {
          setInputData('10');
        } else if (aiResult.algorithm.includes('Knapsack')) {
          setInputData('{"capacity": 10, "weights": [2, 1, 3, 2], "values": [12, 10, 20, 15]}');
        } else if (aiResult.algorithm.includes('LCS')) {
          setInputData('{"str1": "ABCDGH", "str2": "AEDFHR"}');
        }
      } else if (aiResult.dataType === 'tree') {
        setInputData('{"value": 50, "left": {"value": 30, "left": {"value": 20}, "right": {"value": 40}}, "right": {"value": 70, "left": {"value": 60}, "right": {"value": 80}}}');
      }
    } catch (error) {
      setAiError(error.message);
      const fallbackResult = fallbackPatternAnalysis(codeText);
      if (fallbackResult) {
        setDetectedAlgorithm(fallbackResult);
        setConfidence(fallbackResult.confidence);
        if (fallbackResult.dataType === 'graph') {
          setInputData('{"nodes": ["A", "B", "C", "D", "E"], "edges": [["A", "B"], ["A", "C"], ["B", "D"], ["C", "E"], ["D", "E"]], "startNode": "A"}');
        } else if (fallbackResult.dataType === 'weighted_graph') {
          setInputData('{"nodes": ["A", "B", "C", "D", "E"], "edges": [["A", "B", 4], ["A", "C", 2], ["B", "D", 3], ["C", "E", 1], ["D", "E", 5]], "startNode": "A"}');
        } else if (fallbackResult.dataType === 'array') {
          setInputData('[64, 34, 25, 12, 22, 11, 90]');
        } else if (fallbackResult.dataType === 'dp') {
          if (fallbackResult.algorithm.includes('Fibonacci')) {
            setInputData('10');
          } else if (fallbackResult.algorithm.includes('Knapsack')) {
            setInputData('{"capacity": 10, "weights": [2, 1, 3, 2], "values": [12, 10, 20, 15]}');
          } else if (fallbackResult.algorithm.includes('LCS')) {
            setInputData('{"str1": "ABCDGH", "str2": "AEDFHR"}');
          }
        } else if (fallbackResult.dataType === 'tree') {
          setInputData('{"value": 50, "left": {"value": 30, "left": {"value": 20}, "right": {"value": 40}}, "right": {"value": 70, "left": {"value": 60}, "right": {"value": 80}}}');
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
      if (dataType === 'dp') {
        // Handle different DP input formats
        if (input.includes('{')) {
          return JSON.parse(input);
        } else {
          // For simple numeric inputs like Fibonacci
          const num = parseInt(input.trim());
          if (isNaN(num)) {
            throw new Error('Invalid number');
          }
          return num;
        }
      } else if (dataType === 'graph' || dataType === 'weighted_graph') {
        const parsed = JSON.parse(input);
        // Ensure we have required properties with defaults
        return { 
          nodes: parsed.nodes || [],
          edges: parsed.edges || [],
          startNode: parsed.startNode || (parsed.nodes && parsed.nodes[0]) || 'A'
        };
      } else if (dataType === 'tree') {
        return JSON.parse(input);
      } else if (dataType === 'array') {
        const parsed = JSON.parse(input);
        return Array.isArray(parsed) ? parsed : parsed.array || [];
      } else {
        // Default to array parsing
        const parsed = JSON.parse(input);
        if (Array.isArray(parsed)) {
          return parsed;
        } else if (parsed.array) {
          return parsed.array;
        } else if (parsed.target) {
          // Binary search format
          return parsed;
        }
        return [];
      }
    } catch (error) {
      console.error('Error parsing input data:', error);
      
      // Return safe defaults based on dataType
      if (dataType === 'dp') {
        // For DP, try to parse as simple number first
        const num = parseInt(input);
        if (!isNaN(num)) {
          return num;
        }
        return 10; // Default for Fibonacci
      } else if (dataType === 'graph' || dataType === 'weighted_graph') {
        return {
          nodes: ['A', 'B', 'C', 'D', 'E'],
          edges: [['A', 'B'], ['A', 'C'], ['B', 'D'], ['C', 'E'], ['D', 'E']],
          startNode: 'A'
        };
      }
      return [64, 34, 25, 12, 22, 11, 90]; // Default array
    }
  }, []);

  // Generate steps
  const generateSteps = (data, algorithm) => {
    if (!algorithm) return [];
    
    if (algorithm.type === 'graph') {
      if (algorithm.algorithm.includes('Dijkstra')) {
        return generateDijkstraSteps(data);
      } else if (algorithm.algorithm.includes('DFS')) {
        return generateDFSSteps(data);
      } else {
        return generateBFSSteps(data);
      }
    } else if (algorithm.type === 'sorting') {
      if (algorithm.algorithm.includes('Quick')) {
        return generateQuickSortSteps(data);
      } else if (algorithm.algorithm.includes('Merge')) {
        return generateMergeSortSteps(data);
      } else {
        return generateSortSteps(data);
      }
    } else if (algorithm.type === 'searching') {
      return generateBinarySearchSteps(data);
    } else if (algorithm.type === 'dynamic_programming') {
      if (algorithm.algorithm.includes('Fibonacci')) {
        return generateFibonacciSteps(data);
      } else if (algorithm.algorithm.includes('Knapsack')) {
        return generateKnapsackSteps(data);
      } else if (algorithm.algorithm.includes('LCS')) {
        return generateLCSSteps(data);
      }
    } else if (algorithm.type === 'tree') {
      return generateTreeTraversalSteps(data);
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
      alert('No algorithm detected! Try writing a more complete algorithm.');
      return;
    }

    const data = parseInputData(inputData, detectedAlgorithm.dataType);
    if (data === null || data === undefined) {
      alert('Invalid input data format! Please check your input.');
      return;
    }

    // Validate data based on algorithm type
    if (detectedAlgorithm.type === 'graph') {
      if (!data.nodes || !Array.isArray(data.nodes) || data.nodes.length === 0) {
        alert('Invalid graph data! Please provide nodes array.');
        return;
      }
      if (!data.edges || !Array.isArray(data.edges)) {
        alert('Invalid graph data! Please provide edges array.');
        return;
      }
    } else if (detectedAlgorithm.type === 'sorting') {
      if (!Array.isArray(data) || data.length === 0) {
        alert('Invalid array data! Please provide a non-empty array.');
        return;
      }
    } else if (detectedAlgorithm.type === 'dynamic_programming') {
      if (detectedAlgorithm.algorithm.includes('Fibonacci')) {
        if (typeof data !== 'number' || data < 0 || data > 50) {
          alert('Invalid Fibonacci input! Please provide a number between 0 and 50.');
          return;
        }
      } else if (detectedAlgorithm.algorithm.includes('Knapsack')) {
        if (!data.capacity || !data.weights || !data.values) {
          alert('Invalid Knapsack input! Please provide capacity, weights, and values.');
          return;
        }
      } else if (detectedAlgorithm.algorithm.includes('LCS')) {
        if (!data.str1 || !data.str2) {
          alert('Invalid LCS input! Please provide str1 and str2.');
          return;
        }
      }
    }

    try {
      const generatedSteps = generateSteps(data, detectedAlgorithm);
      if (generatedSteps.length === 0) {
        alert('No visualization steps generated. Please check your algorithm and input data.');
        return;
      }
      
      setSteps(generatedSteps);
      setCurrentStep(0);
      setIsPlaying(false);
    } catch (error) {
      console.error('Error generating visualization steps:', error);
      alert(`Error generating visualization: ${error.message}. Please check your input data format.`);
    }
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
    return () => {
      clearInterval(intervalRef.current);
      clearInterval(realtimeRef.current);
    };
  }, []);

  // Theme styles
  const theme = {
    bg: isDarkMode ? 'bg-gray-900' : 'bg-white',
    card: isDarkMode ? 'bg-gray-950' : 'bg-white',
    text: isDarkMode ? 'text-gray-100' : 'text-gray-900',
    textMuted: isDarkMode ? 'text-gray-400' : 'text-gray-500',
    border: isDarkMode ? 'border-gray-800' : 'border-gray-300',
    editor: isDarkMode ? 'bg-gray-900' : 'bg-gray-50',
    button: isDarkMode
      ? 'bg-gray-800 hover:bg-gray-700'
      : 'bg-blue-50 hover:bg-blue-100 border-blue-200 text-blue-900'
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
    } else if (visualState.type === 'weighted_graph') {
      return renderWeightedGraph();
    } else if (visualState.type === 'sorting') {
      return renderSorting();
    } else if (visualState.type === 'searching') {
      return renderSearching();
    } else if (visualState.type === 'dp') {
      return renderDP();
    } else if (visualState.type === 'dp_table') {
      return renderDPTable();
    } else if (visualState.type === 'lcs_table') {
      return renderLCSTable();
    } else if (visualState.type === 'tree') {
      return renderTree();
    }

    return <div className={`h-64 ${theme.card} rounded-md border ${theme.border} p-4 flex items-center justify-center`}>Coming soon!</div>;
  };

  const renderDP = () => {
    return (
      <div className={`h-64 ${theme.card} rounded-md border ${theme.border} p-4`}>
        <div className="mb-4 text-center">
          <span className="text-purple-500 font-mono text-lg">Fibonacci Sequence</span>
        </div>
        
        <div className="flex items-center justify-center gap-2 h-32 overflow-x-auto">
          {visualState.dp.map((value, index) => (
            <div key={index} className="flex flex-col items-center">
              <div className={`w-12 h-12 rounded border-2 flex items-center justify-center font-mono text-sm transition-all duration-300 ${
                index === visualState.current 
                  ? 'bg-purple-500 border-purple-400 text-white transform scale-110' 
                  : isDarkMode ? 'bg-gray-700 border-gray-600 text-gray-300' : 'bg-gray-200 border-gray-300 text-gray-700'
              }`}>
                {value}
              </div>
              <div className={`text-xs mt-1 ${theme.textMuted}`}>F({index})</div>
            </div>
          ))}
        </div>
        
        <div className="mt-4 text-center">
          <span className="text-purple-500 font-mono text-sm">
            Current: F({visualState.current}) = {visualState.dp[visualState.current]}
          </span>
        </div>
      </div>
    );
  };

  const renderDPTable = () => {
    const { dp, current, items } = visualState;
    
    return (
      <div className={`h-64 ${theme.card} rounded-md border ${theme.border} p-4 overflow-auto`}>
        <div className="mb-2 text-center">
          <span className="text-purple-500 font-mono text-sm">0/1 Knapsack DP Table</span>
        </div>
        
        <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${dp[0].length + 1}, minmax(0, 1fr))` }}>
          {/* Header row */}
          <div className="text-xs text-center font-bold">W/I</div>
          {dp[0].map((_, colIndex) => (
            <div key={colIndex} className="text-xs text-center font-bold">{colIndex}</div>
          ))}
          
          {/* Data rows */}
          {dp.map((row, rowIndex) => (
            <React.Fragment key={rowIndex}>
              <div className="text-xs text-center font-bold">{rowIndex}</div>
              {row.map((cell, colIndex) => (
                <div key={colIndex} className={`text-xs text-center p-1 border rounded ${
                  current.i === rowIndex && current.w === colIndex
                    ? 'bg-purple-500 text-white'
                    : isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-100 border-gray-300'
                }`}>
                  {cell}
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
      </div>
    );
  };

  const renderLCSTable = () => {
    const { dp, str1, str2, current } = visualState;
    
    return (
      <div className={`h-64 ${theme.card} rounded-md border ${theme.border} p-4 overflow-auto`}>
        <div className="mb-2 text-center">
          <span className="text-blue-500 font-mono text-sm">LCS: "{str1}" vs "{str2}"</span>
        </div>
        
        <div className="grid gap-1 text-xs" style={{ gridTemplateColumns: `repeat(${str2.length + 2}, minmax(0, 1fr))` }}>
          {/* Header */}
          <div></div>
          <div></div>
          {str2.split('').map((char, i) => (
            <div key={i} className="text-center font-bold">{char}</div>
          ))}
          
          {/* Rows */}
          {dp.map((row, i) => (
            <React.Fragment key={i}>
              {i === 0 ? <div></div> : <div className="text-center font-bold">{str1[i-1]}</div>}
              {row.map((cell, j) => (
                <div key={j} className={`text-center p-1 border rounded ${
                  current.i === i && current.j === j
                    ? 'bg-blue-500 text-white'
                    : isDarkMode ? 'bg-gray-700 border-gray-600' : 'bg-gray-100 border-gray-300'
                }`}>
                  {cell}
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
      </div>
    );
  };

  const renderWeightedGraph = () => {
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
        <div className="mb-2 text-center">
          <span className="text-orange-500 font-mono text-sm">Dijkstra's Shortest Path</span>
        </div>
        
        <div className="flex justify-center">
          <svg width="300" height="150" viewBox="0 0 300 150">
            {/* Edges with weights */}
            {edges.map(([from, to, weight], i) => (
              <g key={i}>
                <line x1={nodePositions[from]?.x} y1={nodePositions[from]?.y}
                      x2={nodePositions[to]?.x} y2={nodePositions[to]?.y}
                      stroke={isDarkMode ? "#6b7280" : "#9ca3af"} strokeWidth="2" />
                <text x={(nodePositions[from]?.x + nodePositions[to]?.x) / 2}
                      y={(nodePositions[from]?.y + nodePositions[to]?.y) / 2 - 5}
                      textAnchor="middle" fill="orange" fontSize="12" fontWeight="bold">
                  {weight}
                </text>
              </g>
            ))}
            
            {/* Nodes with distances */}
            {nodes.map(node => {
              const isVisited = visualState.visited?.has(node);
              const isCurrent = visualState.current === node;
              const distance = visualState.distances?.[node];
              
              let nodeColor = isDarkMode ? '#374151' : '#d1d5db';
              let textColor = isDarkMode ? '#d1d5db' : '#374151';
              
              if (isCurrent) {
                nodeColor = '#f97316';
                textColor = '#ffffff';
              } else if (isVisited) {
                nodeColor = '#16a34a';
                textColor = '#ffffff';
              }
              
              return (
                <g key={node}>
                  <circle cx={nodePositions[node]?.x} cy={nodePositions[node]?.y} r="20"
                          fill={nodeColor} stroke={isDarkMode ? "#4b5563" : "#9ca3af"} strokeWidth="2" />
                  <text x={nodePositions[node]?.x} y={nodePositions[node]?.y}
                        textAnchor="middle" fill={textColor} fontSize="12" fontWeight="bold">
                    {node}
                  </text>
                  <text x={nodePositions[node]?.x} y={nodePositions[node]?.y + 35}
                        textAnchor="middle" fill="orange" fontSize="10" fontWeight="bold">
                    {distance === Infinity ? '∞' : distance}
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>
    );
  };

  const renderSearching = () => {
    const { array, target, left, right, mid, found } = visualState;
    
    return (
      <div className={`h-64 ${theme.card} rounded-md border ${theme.border} p-4`}>
        <div className="mb-4 text-center">
          <span className="text-sm">Target: </span>
          <span className="text-yellow-500 font-mono font-bold">{target}</span>
        </div>
        
        <div className="flex items-center justify-center gap-1 h-32">
          {array.map((value, index) => {
            let bgColor = isDarkMode ? 'bg-gray-700' : 'bg-gray-200';
            
            if (found === index) {
              bgColor = 'bg-green-500';
            } else if (mid === index) {
              bgColor = 'bg-yellow-500';
            } else if (left !== undefined && right !== undefined) {
              if (index >= left && index <= right) {
                bgColor = 'bg-blue-500';
              } else {
                bgColor = isDarkMode ? 'bg-gray-800' : 'bg-gray-300';
              }
            }
            
            return (
              <div key={index} className="flex flex-col items-center">
                <div className={`${bgColor} w-10 h-10 rounded flex items-center justify-center text-white font-mono text-sm transition-all duration-200`}>
                  {value}
                </div>
                <div className={`text-xs ${theme.textMuted} mt-1`}>{index}</div>
              </div>
            );
          })}
        </div>
        
        {left !== undefined && right !== undefined && (
          <div className="mt-4 flex justify-center gap-6 text-sm font-mono">
            <span className="text-blue-400">Left: {left}</span>
            <span className="text-yellow-400">Mid: {mid ?? 'N/A'}</span>
            <span className="text-red-400">Right: {right}</span>
          </div>
        )}
      </div>
    );
  };

  const renderTree = () => {
    const renderTreeNode = (node, x, y, level = 0) => {
      if (!node) return null;
      
      const isVisiting = visualState.visiting === node.value;
      const isProcessing = visualState.processing === node.value;
      
      let nodeColor = isDarkMode ? '#374151' : '#d1d5db';
      let textColor = isDarkMode ? '#d1d5db' : '#374151';
      
      if (isProcessing) {
        nodeColor = '#16a34a';
        textColor = '#ffffff';
      } else if (isVisiting) {
        nodeColor = '#eab308';
        textColor = '#111827';
      }
      
      const leftX = x - (80 / (level + 1));
      const rightX = x + (80 / (level + 1));
      const childY = y + 60;

      return (
        <g key={node.value}>
          {/* Edges to children */}
          {node.left && (
            <line x1={x} y1={y} x2={leftX} y2={childY} stroke={isDarkMode ? "#6b7280" : "#9ca3af"} strokeWidth="2" />
          )}
          {node.right && (
            <line x1={x} y1={y} x2={rightX} y2={childY} stroke={isDarkMode ? "#6b7280" : "#9ca3af"} strokeWidth="2" />
          )}
          
          {/* Node */}
          <circle cx={x} cy={y} r="20" fill={nodeColor} stroke={isDarkMode ? "#4b5563" : "#9ca3af"} strokeWidth="2" />
          <text x={x} y={y + 5} textAnchor="middle" fill={textColor} fontSize="14" fontWeight="bold">
            {node.value}
          </text>
          
          {/* Recursive calls for children */}
          {node.left && renderTreeNode(node.left, leftX, childY, level + 1)}
          {node.right && renderTreeNode(node.right, rightX, childY, level + 1)}
        </g>
      );
    };

    return (
      <div className={`h-64 ${theme.card} rounded-md border ${theme.border} p-4`}>
        <div className="mb-2 text-center">
          <span className="text-green-500 font-mono text-sm">Tree Traversal (Inorder)</span>
        </div>
        
        <div className="flex justify-center">
          <svg width="300" height="200" viewBox="0 0 300 200">
            {renderTreeNode(visualState.tree, 150, 30)}
          </svg>
        </div>
        
        <div className="mt-2 text-center">
          <span className="text-green-500">Result: </span>
          <span className="text-green-400 font-mono">[{visualState.result?.join(', ') || ''}]</span>
        </div>
      </div>
    );
  };

  const renderGraph = () => {
    // Validate visualState.graph
    if (!visualState.graph || !visualState.graph.nodes || !Array.isArray(visualState.graph.nodes)) {
      return (
        <div className={`h-64 ${theme.card} rounded-md border ${theme.border} p-4 flex items-center justify-center`}>
          <span className={theme.textMuted}>Invalid graph data</span>
        </div>
      );
    }

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
            {/* Render edges if they exist */}
            {edges && Array.isArray(edges) && edges.map(([from, to], i) => (
              <line key={i} x1={nodePositions[from]?.x} y1={nodePositions[from]?.y}
                    x2={nodePositions[to]?.x} y2={nodePositions[to]?.y}
                    stroke={isDarkMode ? "#6b7280" : "#9ca3af"} strokeWidth="2" />
            ))}
            
            {/* Render nodes */}
            {nodes.map(node => {
              const isVisited = visualState.visited?.has(node);
              const isCurrent = visualState.current === node;
              const inQueue = visualState.queue?.includes(node);
              const inStack = visualState.stack?.includes(node);
              
              let nodeColor = isDarkMode ? '#374151' : '#d1d5db';
              let textColor = isDarkMode ? '#d1d5db' : '#374151';
              
              if (isCurrent) {
                nodeColor = '#eab308';
                textColor = '#111827';
              } else if (isVisited) {
                nodeColor = '#16a34a';
                textColor = '#ffffff';
              } else if (inQueue || inStack) {
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
          {visualState.stack && (
            <>
              <span className="ml-4 text-purple-500">Stack: </span>
              <span className="text-purple-400 font-mono">[{visualState.stack?.join(', ') || ''}]</span>
            </>
          )}
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
                <h1 className="text-xl font-semibold">Algojuice</h1>
              </div>
              <span className={`${theme.button} border ${theme.border} px-2 py-1 rounded-full text-xs`}>Public</span>
              {tokensUsed > 0 && (
                <span className="bg-blue-900 text-blue-300 px-2 py-1 rounded-full text-xs">{tokensUsed} tokens</span>
              )}
            </div>
            {/* Theme Switcher */}
            <div className="flex items-center gap-3">
              <button
                className={`p-2 rounded-full border ${theme.border} ${theme.button}`}
                title={isDarkMode ? "Switch to Light Mode" : "Switch to Dark Mode"}
                onClick={() => setIsDarkMode((prev) => !prev)}
              >
                {isDarkMode ? <Sun className="w-5 h-5 text-yellow-400" /> : <Moon className="w-5 h-5 text-gray-700" />}
              </button>
              <span className={`italic text-sm ${theme.textMuted}`}>
                "Throughout heaven and earth, I alone am the honored one." – Gojo Satoru
              </span>
            </div>
          </div>
        </div>
      </header>

      <div className="container mx-auto p-6">
        {/* Navigation */}
        <div className={`border-b ${theme.border} mb-6`}>
          <nav className="flex gap-6">
            <button
              className={`flex items-center gap-2 px-3 py-2 border-b-2 border-orange-500 ${theme.text} font-medium`}
              onClick={() => {
                if (codeInputRef.current) {
                  codeInputRef.current.focus();
                }
              }}
            >
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
            {/* AI Detection */}
            <div className={`border-b ${theme.border} px-4 py-3 ${theme.editor}`}>
              <div className="flex items-center gap-2 mb-2">
                <Brain className={`w-4 h-4 ${isAnalyzing ? 'animate-pulse text-yellow-500' : 'text-blue-500'}`} />
                <span className={`text-sm font-medium ${theme.textMuted}`}>
                  {isAnalyzing
                    ? 'Juicing your code...'
                    : 'Your code will be juiced and visualized for you!'}
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
                      ref={codeInputRef}
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

        {/* Visualization Types Info moved to bottom */}
        <div className={`mt-10 ${theme.card} rounded-md border ${theme.border} p-4`}>
          <h2 className="text-lg font-semibold mb-2">Available Visualizations</h2>
          <ul className={`list-disc pl-6 text-sm ${theme.textMuted}`}>
            <li>Graph Traversal: Breadth-First Search (BFS), Depth-First Search (DFS), Dijkstra's Algorithm</li>
            <li>Sorting: Bubble Sort, Quick Sort, Merge Sort</li>
            <li>Searching: Binary Search</li>
            <li>Dynamic Programming: Fibonacci Sequence, 0/1 Knapsack, Longest Common Subsequence (LCS)</li>
            <li>Tree Traversal: Inorder Traversal</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AlgorithmVisualizer;