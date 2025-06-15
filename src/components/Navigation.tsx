
import React from 'react';
import { Button } from '@/components/ui/button';
import { Microscope, TestTube, Atom, FlaskConical, Calculator, Computer } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Navigation = () => {
  const navigate = useNavigate();

  const handleLogoClick = () => {
    navigate('/');
  };

  return (
    <nav className="bg-black/95 backdrop-blur-sm border-b border-gray-800 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div 
            className="flex items-center space-x-2 cursor-pointer hover:opacity-80 transition-opacity"
            onClick={handleLogoClick}
          >
            <Microscope className="w-8 h-8 text-orange-500" />
            <span className="text-xl font-bold text-white">ScienceHub</span>
          </div>
          
          <div className="hidden md:flex items-center space-x-8">
            <a href="#discoveries" className="text-gray-300 hover:text-orange-500 transition-colors">Discoveries</a>
            <a href="#experiments" className="text-gray-300 hover:text-orange-500 transition-colors">Experiments</a>
            <a href="#latex" className="text-gray-300 hover:text-orange-500 transition-colors">Physics LaTeX</a>
            <a href="#math" className="text-gray-300 hover:text-orange-500 transition-colors">Math</a>
            <a href="#computer-science" className="text-gray-300 hover:text-orange-500 transition-colors">Computer Science</a>
            <a href="/notes" className="text-gray-300 hover:text-orange-500 transition-colors">Notes</a>
            <a href="/contact" className="text-gray-300 hover:text-orange-500 transition-colors">Contact</a>
          </div>
          
          <Button className="bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white border-0">
            Get Started
          </Button>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
