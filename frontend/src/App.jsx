import { useState, useRef, useCallback } from 'react'
import './App.css'

const API_URL = 'http://127.0.0.1:8000'

function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)

  // ── Handle file selection ──────────────────────────────────────────────
  const handleFile = useCallback((file) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      setError('Please upload an image file (JPEG, PNG, WEBP)')
      return
    }
    if (file.size > 10 * 1024 * 1024) {
      setError('File too large. Please upload an image under 10MB.')
      return
    }
    setSelectedImage(file)
    setPreviewUrl(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }, [])

  const handleFileInput = (e) => {
    handleFile(e.target.files[0])
  }

  // ── Drag & Drop ────────────────────────────────────────────────────────
  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  // ── API Call ───────────────────────────────────────────────────────────
  const analyzeFruit = async () => {
    if (!selectedImage) return

    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append('file', selectedImage)

      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errData = await response.json().catch(() => null)
        throw new Error(errData?.detail || `Server error (${response.status})`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      console.error('Prediction error:', err)
      if (err.message.includes('Failed to fetch') || err.message.includes('NetworkError')) {
        setError('Cannot connect to the server. Make sure the FastAPI backend is running on port 8000.')
      } else {
        setError(err.message || 'Something went wrong. Please try again.')
      }
    } finally {
      setLoading(false)
    }
  }

  // ── Clean fruit name (remove variant numbers) ─────────────────────────
  const cleanFruitName = (name) => {
    return name.replace(/\s*\d+\s*$/, '').trim()
  }

  // ── Render ─────────────────────────────────────────────────────────────
  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header__inner">
          <div className="header__brand">
            <span className="header__icon">🍎</span>
            <h1 className="header__title">FruitLens</h1>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="main">
        {/* Hero */}
        <section className="hero">
          <h2 className="hero__title">
            Know Your Fruits{' '}
            <span className="hero__title-accent">Instantly</span>
          </h2>
          <p className="hero__subtitle">
            Snap or upload a fruit photo to instantly identify it and discover
            its nutritional value, health benefits, and fun facts.
          </p>
        </section>

        {/* Upload Zone */}
        <div
          id="upload-zone"
          className={`upload-zone ${isDragging ? 'upload-zone--dragging' : ''} ${previewUrl ? 'upload-zone--has-image' : ''}`}
          onClick={() => fileInputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            id="file-input"
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="upload-zone__input"
            onChange={handleFileInput}
          />

          {previewUrl ? (
            <div className="preview">
              <img
                src={previewUrl}
                alt="Selected fruit"
                className="preview__image"
              />
              <span className="preview__change">Click or drag to change image</span>
            </div>
          ) : (
            <>
              <div className="upload-zone__icon">📸</div>
              <p className="upload-zone__text">
                Drop your fruit image here or click to browse
              </p>
              <p className="upload-zone__subtext">
                Supports JPEG, PNG, WEBP (max 10MB)
              </p>
            </>
          )}
        </div>

        {/* Analyze Button */}
        {selectedImage && (
          <div style={{ textAlign: 'center' }}>
            <button
              id="analyze-btn"
              className="analyze-btn"
              onClick={analyzeFruit}
              disabled={loading}
            >
              {loading ? (
                <>
                  <div className="analyze-btn__spinner" />
                  Analyzing...
                </>
              ) : (
                <>
                  🔍 Analyze Fruit
                </>
              )}
            </button>
          </div>
        )}

        {/* Error */}
        {error && (
          <div id="error-message" className="error">
            ⚠️ {error}
          </div>
        )}

        {/* Results */}
        {result && (
          <section className="results" id="results-section">
            <div className="results__grid">
              {/* Prediction Card */}
              <div className="card prediction">
                <div className="card__header">
                  <span className="card__icon">🎯</span>
                  <span className="card__title">Prediction</span>
                </div>
                <h3 className="prediction__name">
                  {cleanFruitName(result.prediction.fruit_name)}
                </h3>
                
                {result.prediction.is_refined && (
                  <div className="prediction__refined-badge" title="Verified by Gemini Vision for higher accuracy">
                    ✨ AI Refined
                  </div>
                )}

                <p className="prediction__confidence">
                  {result.prediction.is_refined ? 'Confidence: Very High' : `Confidence: ${result.prediction.confidence}%`}
                </p>

                <div className="top-predictions">
                  <p className="top-predictions__label">PyTorch Model Guesses:</p>
                  {(result.prediction.pytorch_guesses || []).map((pred, i) => (
                    <div key={i} className="top-pred">
                      <span className="top-pred__rank">{i + 1}</span>
                      <span className="top-pred__name">
                        {cleanFruitName(pred.name)}
                      </span>
                      <div className="top-pred__bar-wrapper">
                        <div
                          className="top-pred__bar"
                          style={{ width: `${Math.max(pred.confidence, 0.5)}%` }}
                        />
                      </div>
                      <span className="top-pred__value">{pred.confidence}%</span>
                    </div>
                  ))}
                </div>

              </div>

              {/* Description Card */}
              {result.ai_description && (
                <div className="card description">
                  <div className="card__header">
                    <span className="card__icon">📝</span>
                    <span className="card__title">About this Fruit</span>
                  </div>
                  <p className="description__text">
                    {result.ai_description.description}
                  </p>
                </div>
              )}

              {/* Nutrition Card */}
              {result.ai_description?.nutrition && (
                <div className="card nutrition">
                  <div className="card__header">
                    <span className="card__icon">🥗</span>
                    <span className="card__title">Nutrition (per 100g)</span>
                  </div>
                  <p className="nutrition__text">
                    {result.ai_description.nutrition}
                  </p>
                </div>
              )}

              {/* Health Benefits Card */}
              {result.ai_description?.health_benefits?.length > 0 && (
                <div className="card benefits">
                  <div className="card__header">
                    <span className="card__icon">💪</span>
                    <span className="card__title">Health Benefits</span>
                  </div>
                  <ul className="benefits__list">
                    {result.ai_description.health_benefits.map((benefit, i) => (
                      <li key={i} className="benefits__item">
                        <span className="benefits__check">+</span>
                        {benefit}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Fun Fact Card */}
              {result.ai_description?.fun_fact && (
                <div className="card card--full funfact">
                  <div className="card__header">
                    <span className="card__icon">💡</span>
                    <span className="card__title">Fun Fact</span>
                  </div>
                  <p className="funfact__text">
                    "{result.ai_description.fun_fact}"
                  </p>
                </div>
              )}
            </div>
          </section>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>FruitLens &mdash; Identify fruits, discover nutrition</p>
      </footer>
    </div>
  )
}

export default App
