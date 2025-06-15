
import React from 'react';
import { Button } from '@/components/ui/button';
import { Microscope, TestTube, Atom, FlaskConical } from 'lucide-react';

const Navigation = () => {
  return (
    <nav className="bg-white/95 backdrop-blur-sm border-b border-gray-200 sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-2">
            <Microscope className="w-8 h-8 text-blue-600" />
            <span className="text-xl font-bold text-gray-900">ScienceHub</span>
          </div>
          
          <div className="hidden md:flex items-center space-x-8">
            <a href="#home" className="text-gray-700 hover:text-blue-600 transition-colors">Home</a>
            <a href="#discoveries" className="text-gray-700 hover:text-blue-600 transition-colors">Discoveries</a>
            <a href="#experiments" className="text-gray-700 hover:text-blue-600 transition-colors">Experiments</a>
            <a href="#latex" className="text-gray-700 hover:text-blue-600 transition-colors">Physics LaTeX</a>
            <a href="#contact" className="text-gray-700 hover:text-blue-600 transition-colors">Contact</a>
          </div>
          
          <Button className="bg-blue-600 hover:bg-blue-700">
            Get Started
          </Button>
        </div>
      </div>
    </nav>
  );
};

export default Navigation;
