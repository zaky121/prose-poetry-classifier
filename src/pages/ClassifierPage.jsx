import { useState, useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

export default function ClassifierPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [text, setText] = useState('');
  const [model, setModel] = useState('soberta');
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState([]);
  const [activeModelTab, setActiveModelTab] = useState('soberta');
  const [fetchError, setFetchError] = useState(null);
  
  const [apiStatus] = useState({
    soberta: 'healthy',
    afriberta: 'healthy',
    rnn: 'healthy',
    lstm: 'healthy'
  });

  useEffect(() => {
    const searchParams = new URLSearchParams(location.search);
    const sample = searchParams.get('sample');
    const modelParam = searchParams.get('model');
    
    if (sample) setText(decodeURIComponent(sample));
    if (modelParam && ['soberta', 'afriberta', 'rnn', 'lstm'].includes(modelParam)) {
      setModel(modelParam);
      setActiveModelTab(modelParam);
    }
  }, [location.search]);

  const models = {
    soberta: {
      name: "SoBERTa",
      description: "Somali-optimized BERT model",
      color: "#3a7ca5",
      endpoint: "http://127.0.0.1:8000/predict"
    },
    afriberta: {
      name: "AfriBERTa",
      description: "AfriBERTa fine-tuned model",
      color: "#5a3d85",
      endpoint: "http://127.0.0.1:8001/predict"
    },
    rnn: {
      name: "RNN",
      description: "Recurrent Neural Network",
      color: "#d35400",
      endpoint: "http://127.0.0.1:8002/predict"
    },
    lstm: {
      name: "LSTM",
      description: "Long Short-Term Memory",
      color: "#27ae60",
      endpoint: "http://127.0.0.1:8003/predict",
      min_length: 5
    }
  };

  const classifyText = async () => {
    if (!text.trim()) {
      setFetchError('Text cannot be empty');
      return null;
    }

    if (text.length > 10000) {
      setFetchError('Text exceeds 10,000 character limit');
      return null;
    }

    if (model === 'lstm' && text.split(/\s+/).length < models.lstm.min_length) {
      setFetchError(`LSTM requires minimum ${models.lstm.min_length} words`);
      return null;
    }

    setIsLoading(true);
    setFetchError(null);
    
    try {
      const payload = {
        text: text,
        ...(model === 'lstm' && { min_length: models.lstm.min_length })
      };

      const response = await fetch(models[model].endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || errorData.message || 'Classification failed');
      }
      
      const data = await response.json();
      
      return {
        classification: data.classification || "Unknown",
        processed_text: data.processed_text || text,
        is_poetry: data.is_poetry || false,
        confidence: data.confidence || 0,
        score: data.score || 0,
        model: data.model || models[model].name,
        inference_time: data.inference_time || 0
      };
    } catch (error) {
      console.error('Classification error:', error);
      setFetchError(error.message);
      return null;
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const result = await classifyText();
    if (result) {
      setResult(result);
      addToHistory(result);
      navigate(`?model=${model}`, { replace: true });
    }
  };

  const addToHistory = (result) => {
    setHistory(prev => [
      { 
        id: Date.now(),
        text: text.length > 50 ? `${text.substring(0, 50)}...` : text,
        ...result,
        timestamp: new Date().toLocaleString()
      },
      ...prev.slice(0, 9)
    ]);
  };

  return (
    <div className="classification-container">
      <div className="classifier-header">
        <h2>Somali Poetry/Prose Classifier</h2>
        <p>Using {models[model].name} model</p>
        
        <div className="api-status-container">
          {Object.entries(models).map(([key, modelData]) => (
            <div key={key} className="api-status-item">
              <span className="model-name">{modelData.name}:</span>
              <span className={`api-status ${apiStatus[key]}`}>
                {apiStatus[key].charAt(0).toUpperCase() + apiStatus[key].slice(1)}
              </span>
            </div>
          ))}
        </div>
      </div>
      
      {fetchError && (
        <div className="error-message">
          <div className="error-content">
            <span className="error-icon">⚠️</span>
            <div>{fetchError}</div>
          </div>
          <button onClick={() => setFetchError(null)}>Dismiss</button>
        </div>
      )}
      
      <div className="model-tabs">
        {Object.entries(models).map(([key, modelData]) => (
          <button
            key={key}
            className={activeModelTab === key ? 'active' : ''}
            style={{ backgroundColor: activeModelTab === key ? modelData.color : '#f5f5f5' }}
            onClick={() => {
              setModel(key);
              setActiveModelTab(key);
              navigate(`?model=${key}`, { replace: true });
            }}
            disabled={isLoading}
          >
            {modelData.name}
          </button>
        ))}
      </div>
      
      <div className="classifier-grid">
        <form onSubmit={handleSubmit} className="input-section">
          <div className="form-group">
            <label>Somali Text to Analyze</label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Paste Somali text here..."
              rows={12}
              required
              disabled={isLoading}
            />
            <div className="text-counter">
              {text.length} / 10,000 characters • {text.split(/\s+/).length} words
              {model === 'lstm' && (
                <span className="min-length-warning">
                  (Minimum {models.lstm.min_length} words required)
                </span>
              )}
            </div>
          </div>
          
          <div className="form-actions">
            <button 
              type="submit" 
              disabled={isLoading}
              style={{ backgroundColor: models[model].color }}
            >
              {isLoading ? 'Analyzing...' : 'Classify Text'}
            </button>
            <button 
              type="button" 
              onClick={() => {
                setText('');
                setResult(null);
                setFetchError(null);
              }}
              disabled={isLoading}
            >
              Clear
            </button>
          </div>
        </form>
        
        {result && (
          <div className="result-section" style={{ borderColor: models[model].color }}>
            <div className="result-header">
              <h3>Classification Results</h3>
              <span className="model-pill" style={{ backgroundColor: models[model].color }}>
                {result.model}
              </span>
            </div>
            
            <div className={`result-classification ${result.classification.toLowerCase()}`}>
              {result.classification}
              <span>{result.confidence.toFixed(2)}%</span>
            </div>
            
            <div className="result-details">
              <div className="detail-item">
                <span>Confidence:</span>
                <strong>{result.confidence.toFixed(2)}%</strong>
              </div>
              <div className="detail-item">
                <span>Inference Time:</span>
                <strong>{result.inference_time.toFixed(3)}s</strong>
              </div>
            </div>
          </div>
        )}
        
        <div className="history-section">
          <h3>Analysis History</h3>
          {history.length > 0 ? (
            <ul className="history-list">
              {history.map((item) => (
                <li 
                  key={item.id}
                  onClick={() => {
                    setText(item.text);
                    setModel(item.model.toLowerCase());
                    setActiveModelTab(item.model.toLowerCase());
                    setResult(item);
                  }}
                >
                  <div style={{ color: models[item.model.toLowerCase()].color }}>
                    {item.model}
                  </div>
                  <div>{item.text}</div>
                  <div className={`result ${item.classification.toLowerCase()}`}>
                    {item.classification} ({item.confidence.toFixed(2)}%)
                  </div>
                </li>
              ))}
            </ul>
          ) : (
            <p>No history yet</p>
          )}
        </div>
      </div>
    </div>
  );
}