import { Routes, Route } from 'react-router-dom'
import Header from './components/Header'
import Footer from './components/Footer'
import Navbar from './components/Navbar'
import HomePage from './pages/HomePage'
import ClassifierPage from './pages/ClassifierPage'
import './index.css'

function App() {
  return (
    <div className="app">
      <Header />
      <Navbar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/classify" element={<ClassifierPage />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}

export default App