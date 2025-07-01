import { useState } from 'react';
import { Link } from 'react-router-dom';
import './HomePage.css'; // Create this CSS file for custom styles

export default function HomePage() {
  const [activeTab, setActiveTab] = useState('features');
  const [showDemo, setShowDemo] = useState(false);

  const sampleTexts = [
    {
      title: "Romantic Poetry",
      content: "Shall I compare thee to a summer's day? Thou art more lovely and more temperate...",
      author: "William Shakespeare",
      type: "poetry"
    },
    {
      title: "Scientific Prose",
      content: "The quantum theory revolutionizes our understanding of atomic and subatomic processes...",
      author: "Science Journal",
      type: "prose"
    },
    {
      title: "African Poetry",
      content: "The rhythm of the drums echoes the heartbeat of the continent...",
      author: "African Poet",
      type: "poetry"
    }
  ];

  const modelStats = [
    { name: 'RNN', accuracy: '92%', speed: 'Fast', bestFor: 'Basic classification', color: '#FF6B6B' },
    { name: 'LSTM', accuracy: '94%', speed: 'Medium', bestFor: 'Sequential patterns', color: '#4ECDC4' },
    { name: 'SoBERTa', accuracy: '96%', speed: 'Slow', bestFor: 'Contextual understanding', color: '#45B7D1' },
    { name: 'AfriBERTa', accuracy: '95%', speed: 'Slow', bestFor: 'African language texts', color: '#FFA07A' }
  ];

  return (
    <div className="home-container">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-gradient"></div>
        <div className="hero-content">
          <h1 className="hero-title">
            <span className="title-gradient">Advanced</span> Text Classification
          </h1>
          <p className="hero-subtitle">
            Distinguish poetry from prose with cutting-edge AI models
          </p>
          <div className="hero-buttons">
            <Link to="/classify" className="cta-button primary">
              <span>Try Classifier</span>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </Link>
            <button 
              className={`cta-button secondary ${showDemo ? 'active' : ''}`}
              onClick={() => setShowDemo(!showDemo)}
            >
              {showDemo ? 'Hide Demo' : 'Live Demo'}
              <span className="pulse-dot"></span>
            </button>
          </div>
        </div>
        
        {showDemo && (
          <div className="live-demo-container">
            {modelStats.map((model, index) => (
              <div 
                key={model.name}
                className="demo-model-card"
                style={{ 
                  animationDelay: `${index * 0.1}s`,
                  borderColor: model.color
                }}
              >
                <div className="model-badge" style={{ backgroundColor: model.color }}>
                  {model.name}
                </div>
                <div className="demo-progress">
                  <div 
                    className="progress-bar" 
                    style={{ 
                      width: model.accuracy,
                      backgroundColor: model.color
                    }}
                  ></div>
                </div>
                <div className="demo-stats">
                  <span>Accuracy: {model.accuracy}</span>
                  <span>Speed: {model.speed}</span>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>

      {/* Navigation Tabs */}
      <div className="tabs-container">
        <div className="tabs">
          {['features', 'models', 'examples', 'research'].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`tab-button ${activeTab === tab ? 'active' : ''}`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
              {activeTab === tab && <div className="tab-indicator"></div>}
            </button>
          ))}
        </div>
      </div>

      {/* Main Content Sections */}
      <div className="content-container">
        {/* Features Tab */}
        {activeTab === 'features' && (
          <section className="features-section">
            <h3 className="section-title">
              <span className="title-decoration">Platform</span> Capabilities
            </h3>
            <div className="features-grid">
              {[
                {
                  icon: '🧠',
                  title: 'Multi-Model Analysis',
                  description: 'Compare results across RNN, LSTM, SoBERTa, and AfriBERTa models'
                },
                {
                  icon: '📊',
                  title: 'Detailed Metrics',
                  description: 'Get comprehensive analysis with confidence scores and model-specific insights'
                },
                {
                  icon: '🌍',
                  title: 'Multilingual Support',
                  description: 'Especially optimized for African languages through AfriBERTa'
                },
                {
                  icon: '⏱️',
                  title: 'Performance Tracking',
                  description: 'Compare model speed and accuracy for your specific use case'
                }
              ].map((feature, index) => (
                <div 
                  key={index} 
                  className="feature-card"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="feature-icon">{feature.icon}</div>
                  <h4>{feature.title}</h4>
                  <p>{feature.description}</p>
                  <div className="card-hover-effect"></div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Models Tab */}
        {activeTab === 'models' && (
          <section className="models-section">
            <h3 className="section-title">
              <span className="title-decoration">Available</span> Models
            </h3>
            <div className="models-comparison">
              <div className="model-cards">
                {modelStats.map((model) => (
                  <div 
                    key={model.name}
                    className="model-card"
                    style={{ borderTopColor: model.color }}
                  >
                    <div className="model-header">
                      <h4 style={{ color: model.color }}>{model.name}</h4>
                      <div className="accuracy-badge" style={{ backgroundColor: model.color }}>
                        {model.accuracy}
                      </div>
                    </div>
                    <div className="model-details">
                      <div className="detail-item">
                        <span>Speed:</span>
                        <strong>{model.speed}</strong>
                      </div>
                      <div className="detail-item">
                        <span>Best For:</span>
                        <strong>{model.bestFor}</strong>
                      </div>
                    </div>
                    <Link 
                      to={`/classify?model=${model.name.toLowerCase()}`}
                      className="model-try-button"
                      style={{ backgroundColor: model.color }}
                    >
                      Try {model.name}
                    </Link>
                  </div>
                ))}
              </div>
              
              <div className="model-selection-guide">
                <h4>Model Selection Guide</h4>
                <ul>
                  {modelStats.map((model) => (
                    <li key={model.name}>
                      <span className="model-bullet" style={{ backgroundColor: model.color }}></span>
                      <strong>{model.name}:</strong> {model.bestFor}
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          </section>
        )}

        {/* Examples Tab */}
        {activeTab === 'examples' && (
          <section className="examples-section">
            <h3 className="section-title">
              <span className="title-decoration">Example</span> Texts
            </h3>
            <div className="examples-tabs">
              {['All', 'Poetry', 'Prose', 'African'].map((type) => (
                <button 
                  key={type}
                  className={type === 'All' ? 'active' : ''}
                >
                  {type}
                </button>
              ))}
            </div>
            <div className="sample-cards">
              {sampleTexts.map((text, index) => (
                <div 
                  key={index} 
                  className="sample-card"
                  style={{ animationDelay: `${index * 0.1}s` }}
                >
                  <div className="card-header">
                    <div className={`text-type ${text.type}`}>
                      {text.type.toUpperCase()}
                    </div>
                    <h4>{text.title}</h4>
                    <span className="author">— {text.author}</span>
                  </div>
                  <div className="card-content">
                    <p>"{text.content}"</p>
                  </div>
                  <div className="card-actions">
                    {modelStats.map((model) => (
                      <Link
                        key={model.name}
                        to={`/classify?sample=${encodeURIComponent(text.content)}&model=${model.name.toLowerCase()}`}
                        className="model-button"
                        style={{ 
                          backgroundColor: model.color,
                          '--hover-color': `${model.color}DD`
                        }}
                      >
                        {model.name}
                      </Link>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Research Tab */}
        {activeTab === 'research' && (
          <section className="research-section">
            <h3 className="section-title">
              <span className="title-decoration">Research</span> & Methodology
            </h3>
            <div className="research-content">
              <div className="research-card">
                <div className="research-icon">🔬</div>
                <h4>Model Training</h4>
                <p>All models were trained on a diverse corpus of 50,000+ text samples including:</p>
                <ul>
                  <li>Classic and modern poetry</li>
                  <li>Literary and technical prose</li>
                  <li>African language texts</li>
                </ul>
              </div>
              
              <div className="research-card">
                <div className="research-icon">📈</div>
                <h4>Performance Metrics</h4>
                <div className="metrics-chart">
                  {modelStats.map((model) => (
                    <div key={model.name} className="metric-item">
                      <div className="metric-label">
                        <span className="model-color" style={{ backgroundColor: model.color }}></span>
                        {model.name}
                      </div>
                      <div className="metric-bar-container">
                        <div 
                          className="metric-bar"
                          style={{
                            width: model.accuracy,
                            backgroundColor: model.color
                          }}
                        >
                          <span className="metric-value">{model.accuracy}</span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>
        )}
      </div>

      {/* Final CTA Section */}
      <section className="final-cta">
        <div className="cta-content">
          <h3>Ready to analyze your texts?</h3>
          <p>Experience the power of multiple AI models working together</p>
          <Link to="/classify" className="cta-button large">
            Start Classifying Now
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
              <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </Link>
        </div>
      </section>
    </div>
  );
}