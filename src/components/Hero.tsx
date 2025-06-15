
import React from 'react';
import { Button } from '@/components/ui/button';
import { useNavigate } from 'react-router-dom';
import { Microscope, BarChart3 } from 'lucide-react';

const Hero = () => {
  const navigate = useNavigate();

  const handleWatchGraph = () => {
    navigate('/notes');
  };

  return (
    <section className="relative bg-gradient-to-br from-black via-gray-900 to-black py-20 overflow-hidden">
      <div className="absolute inset-0 bg-gradient-to-r from-orange-500/10 to-red-500/10"></div>
      <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <div className="mb-8">
          <Microscope className="w-16 h-16 text-orange-500 mx-auto mb-6" />
          <h1 className="text-4xl md:text-6xl font-bold text-white mb-6">
            Explore the Universe of
            <span className="bg-gradient-to-r from-orange-500 to-red-500 bg-clip-text text-transparent"> Science</span>
          </h1>
          <p className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto mb-8">
            Discover groundbreaking research, conduct virtual experiments, and unlock the mysteries of physics, mathematics, and computer science.
          </p>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <Button className="bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white border-0 text-lg px-8 py-3">
            Start Exploring
          </Button>
          <Button 
            variant="outline" 
            className="border-orange-500 text-orange-500 hover:bg-orange-500 hover:text-white text-lg px-8 py-3"
            onClick={handleWatchGraph}
          >
            <BarChart3 className="w-5 h-5 mr-2" />
            Watch Graph
          </Button>
        </div>
        
        <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
          <div className="p-6">
            <div className="text-3xl font-bold text-orange-500 mb-2">1000+</div>
            <div className="text-gray-400">Scientific Papers</div>
          </div>
          <div className="p-6">
            <div className="text-3xl font-bold text-orange-500 mb-2">50+</div>
            <div className="text-gray-400">Interactive Experiments</div>
          </div>
          <div className="p-6">
            <div className="text-3xl font-bold text-orange-500 mb-2">24/7</div>
            <div className="text-gray-400">Learning Support</div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
