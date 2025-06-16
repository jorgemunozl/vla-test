
import React, { useState } from 'react';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { BookOpen, Calendar, Trash2, BarChart3, Lightbulb } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface Note {
  id: string;
  title: string;
  content: string;
  date: string;
}

const Notes = () => {
  const navigate = useNavigate();
  const [notes, setNotes] = useState<Note[]>([
    {
      id: '1',
      title: 'Newton\'s Second Law',
      content: 'F = ma\n\nForce equals mass times acceleration. This fundamental principle describes the relationship between the forces acting on a body and its motion due to those forces.',
      date: new Date().toLocaleDateString()
    },
    {
      id: '2',
      title: 'Quantum Mechanics Basics',
      content: 'Key principles:\n- Wave-particle duality\n- Uncertainty principle\n- Superposition\n\nQuantum mechanics describes the behavior of matter and energy at the molecular, atomic, nuclear, and even smaller microscopic levels.',
      date: new Date().toLocaleDateString()
    }
  ]);

  const deleteNote = (id: string) => {
    setNotes(notes.filter(note => note.id !== id));
  };

  const handleVisualizeNotes = () => {
    navigate('/visualize-notes');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-900 via-purple-800 to-purple-600">
      <Navigation />
      
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header Section */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
            <span className="text-orange-500 font-medium uppercase tracking-wide text-sm">Physics Knowledge System</span>
          </div>
          <div className="flex items-center justify-center space-x-2 mb-6">
            <BookOpen className="w-8 h-8 text-orange-500" />
            <h1 className="text-4xl font-bold text-white">Physics Notes Library</h1>
          </div>
          <p className="text-xl text-gray-200 max-w-3xl mx-auto mb-8">
            Your curated collection of physics insights and discoveries. Explore your knowledge and visualize connections between concepts.
          </p>
          
          {/* Visualize Notes Button */}
          {notes.length > 0 && (
            <div className="mb-8">
              <Button 
                onClick={handleVisualizeNotes}
                className="bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white border-0 px-6 py-3 text-lg"
              >
                <BarChart3 className="w-5 h-5 mr-2" />
                Visualize Notes
              </Button>
            </div>
          )}
        </div>

        {/* Notes List */}
        <div className="max-w-4xl mx-auto">
          <div className="mb-6">
            <h2 className="text-2xl font-semibold text-white flex items-center">
              Your Physics Library
              <span className="ml-3 bg-orange-500 text-black px-3 py-1 rounded-full text-sm font-bold">
                {notes.length}
              </span>
            </h2>
            <p className="text-gray-200 mt-2">
              {notes.length === 0 
                ? "Your collection is empty." 
                : `Your knowledge collection contains ${notes.length} note${notes.length !== 1 ? 's' : ''}.`
              }
            </p>
          </div>
          
          <div className="space-y-4">
            {notes.length === 0 ? (
              <Card className="bg-gray-900/50 border-gray-700 border-dashed backdrop-blur-sm">
                <CardContent className="text-center py-16">
                  <Lightbulb className="w-16 h-16 text-gray-500 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-gray-400 mb-2">No Notes Available</h3>
                  <p className="text-gray-500">Your physics notes library is empty.</p>
                </CardContent>
              </Card>
            ) : (
              notes.map((note, index) => (
                <Card key={note.id} className="bg-gray-900/70 border-gray-700 hover:border-orange-500/50 transition-all transform hover:scale-[1.02] backdrop-blur-sm">
                  <CardHeader className="pb-2">
                    <div className="flex justify-between items-start">
                      <div className="flex items-center space-x-2">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        <CardTitle className="text-lg text-white">{note.title}</CardTitle>
                        {index === 0 && (
                          <span className="bg-orange-500 text-black text-xs px-2 py-1 rounded font-bold">
                            FEATURED
                          </span>
                        )}
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => deleteNote(note.id)}
                        className="text-red-400 hover:text-red-300 hover:bg-red-900/20"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                    <div className="flex items-center text-sm text-gray-400">
                      <Calendar className="w-4 h-4 mr-1" />
                      {note.date}
                    </div>
                  </CardHeader>
                  <CardContent>
                    <p className="text-gray-200 whitespace-pre-wrap">{note.content}</p>
                  </CardContent>
                </Card>
              ))
            )}
          </div>
        </div>
      </div>

      <Footer />
    </div>
  );
};

export default Notes;
