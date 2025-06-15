import React, { useState } from 'react';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { BookOpen, Calendar, Trash2, Plus, Edit3, Save, Lightbulb, Zap, BarChart3 } from 'lucide-react';

interface Note {
  id: string;
  title: string;
  content: string;
  date: string;
}

const Notes = () => {
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
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [isFormVisible, setIsFormVisible] = useState(false);

  const addNote = () => {
    if (title.trim() && content.trim()) {
      const newNote: Note = {
        id: Date.now().toString(),
        title: title.trim(),
        content: content.trim(),
        date: new Date().toLocaleDateString()
      };
      setNotes([newNote, ...notes]);
      setTitle('');
      setContent('');
      setIsFormVisible(false);
    }
  };

  const deleteNote = (id: string) => {
    setNotes(notes.filter(note => note.id !== id));
  };

  const handleShowForm = () => {
    setIsFormVisible(true);
  };

  const handleCancelForm = () => {
    setIsFormVisible(false);
    setTitle('');
    setContent('');
  };

  const handleVisualizeNotes = () => {
    console.log('Visualize Notes clicked - functionality to be implemented');
    // TODO: Add visualization functionality
  };

  return (
    <div className="min-h-screen bg-black">
      <Navigation />
      
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Header Section with Dynamic Explanation */}
        <div className="text-center mb-12">
          <div className="flex items-center justify-center space-x-2 mb-4">
            <div className="w-2 h-2 bg-orange-500 rounded-full animate-pulse"></div>
            <span className="text-orange-500 font-medium uppercase tracking-wide text-sm">Dynamic Knowledge System</span>
          </div>
          <div className="flex items-center justify-center space-x-2 mb-6">
            <BookOpen className="w-8 h-8 text-orange-500" />
            <h1 className="text-4xl font-bold text-white">Interactive Physics Notes</h1>
          </div>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto mb-8">
            Your personal physics laboratory notebook that grows with your discoveries. Add, organize, and manage your scientific insights in real-time.
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
          
          {/* Dynamic Features Explanation */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
            <Card className="bg-gray-900/50 border-gray-800 hover:border-orange-500/30 transition-all">
              <CardContent className="p-6 text-center">
                <Plus className="w-8 h-8 text-orange-500 mx-auto mb-3" />
                <h3 className="text-white font-semibold mb-2">Add Instantly</h3>
                <p className="text-gray-400 text-sm">Create new notes with formulas, theories, and observations. Your content is saved automatically.</p>
              </CardContent>
            </Card>
            <Card className="bg-gray-900/50 border-gray-800 hover:border-orange-500/30 transition-all">
              <CardContent className="p-6 text-center">
                <Zap className="w-8 h-8 text-orange-500 mx-auto mb-3" />
                <h3 className="text-white font-semibold mb-2">Live Updates</h3>
                <p className="text-gray-400 text-sm">Notes update in real-time as you type. See your counter change and content organize dynamically.</p>
              </CardContent>
            </Card>
            <Card className="bg-gray-900/50 border-gray-800 hover:border-orange-500/30 transition-all">
              <CardContent className="p-6 text-center">
                <Edit3 className="w-8 h-8 text-orange-500 mx-auto mb-3" />
                <h3 className="text-white font-semibold mb-2">Easy Management</h3>
                <p className="text-gray-400 text-sm">Delete unwanted notes with one click. Your notebook stays organized and relevant.</p>
              </CardContent>
            </Card>
          </div>
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Dynamic Add Note Section */}
          <div>
            <div className="mb-6">
              <h2 className="text-2xl font-semibold text-white mb-2">
                Create New Note
              </h2>
              <p className="text-gray-400">
                Watch the counter update as you add notes! Currently showing {notes.length} note{notes.length !== 1 ? 's' : ''}.
              </p>
            </div>

            {!isFormVisible ? (
              <Card className="bg-gray-900 border-gray-800 hover:border-orange-500/50 transition-all cursor-pointer" onClick={handleShowForm}>
                <CardContent className="p-12 text-center">
                  <Plus className="w-12 h-12 text-orange-500 mx-auto mb-4" />
                  <h3 className="text-xl font-semibold text-white mb-2">Add Your First Note</h3>
                  <p className="text-gray-400">Click here to start writing your physics discoveries</p>
                </CardContent>
              </Card>
            ) : (
              <Card className="bg-gray-900 border-orange-500/50">
                <CardHeader>
                  <CardTitle className="text-white flex items-center">
                    <Edit3 className="w-5 h-5 mr-2" />
                    New Physics Note
                  </CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div>
                    <Label htmlFor="title" className="text-gray-300">Title</Label>
                    <Input 
                      id="title"
                      placeholder="E.g., 'Einstein's Mass-Energy Equivalence'"
                      value={title}
                      onChange={(e) => setTitle(e.target.value)}
                      className="bg-gray-800 border-gray-700 text-white placeholder:text-gray-400 focus:border-orange-500"
                    />
                  </div>
                  <div>
                    <Label htmlFor="content" className="text-gray-300">Content</Label>
                    <Textarea 
                      id="content"
                      placeholder="E.g., 'E=mc²&#10;&#10;This equation shows that mass and energy are interchangeable. A small amount of mass can be converted into a tremendous amount of energy...'"
                      value={content}
                      onChange={(e) => setContent(e.target.value)}
                      className="min-h-[200px] bg-gray-800 border-gray-700 text-white placeholder:text-gray-400 focus:border-orange-500"
                    />
                  </div>
                  <div className="flex space-x-3">
                    <Button 
                      onClick={addNote} 
                      className="flex-1 bg-gradient-to-r from-orange-500 to-red-500 hover:from-orange-600 hover:to-red-600 text-white border-0"
                      disabled={!title.trim() || !content.trim()}
                    >
                      <Save className="w-4 h-4 mr-2" />
                      Save Note
                    </Button>
                    <Button 
                      onClick={handleCancelForm}
                      variant="outline"
                      className="border-gray-600 text-gray-300 hover:bg-gray-800"
                    >
                      Cancel
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>

          {/* Dynamic Notes List */}
          <div>
            <div className="mb-6">
              <h2 className="text-2xl font-semibold text-white flex items-center">
                Your Physics Library
                <span className="ml-3 bg-orange-500 text-black px-3 py-1 rounded-full text-sm font-bold">
                  {notes.length}
                </span>
              </h2>
              <p className="text-gray-400 mt-2">
                {notes.length === 0 
                  ? "Your collection is empty. Add your first note to see the magic happen!" 
                  : `Your knowledge grows! Each note you add appears here instantly.`
                }
              </p>
            </div>
            
            <div className="space-y-4 max-h-[600px] overflow-y-auto">
              {notes.length === 0 ? (
                <Card className="bg-gray-900 border-gray-800 border-dashed">
                  <CardContent className="text-center py-16">
                    <Lightbulb className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                    <h3 className="text-xl font-semibold text-gray-500 mb-2">Ready for Your Ideas</h3>
                    <p className="text-gray-600 mb-4">This space will dynamically populate with your notes</p>
                    <div className="text-sm text-gray-700">
                      ← Start by clicking "Add New Note" on the left
                    </div>
                  </CardContent>
                </Card>
              ) : (
                notes.map((note, index) => (
                  <Card key={note.id} className="bg-gray-900 border-gray-800 hover:border-orange-500/50 transition-all transform hover:scale-[1.02]">
                    <CardHeader className="pb-2">
                      <div className="flex justify-between items-start">
                        <div className="flex items-center space-x-2">
                          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                          <CardTitle className="text-lg text-white">{note.title}</CardTitle>
                          {index === 0 && (
                            <span className="bg-orange-500 text-black text-xs px-2 py-1 rounded font-bold">
                              NEWEST
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
                      <div className="flex items-center text-sm text-gray-500">
                        <Calendar className="w-4 h-4 mr-1" />
                        {note.date}
                      </div>
                    </CardHeader>
                    <CardContent>
                      <p className="text-gray-300 whitespace-pre-wrap">{note.content}</p>
                    </CardContent>
                  </Card>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Live Demo Section */}
        {notes.length > 0 && (
          <div className="mt-12 text-center">
            <Card className="bg-gradient-to-r from-orange-500/10 to-red-500/10 border-orange-500/30">
              <CardContent className="p-8">
                <Zap className="w-8 h-8 text-orange-500 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-white mb-2">See the Dynamic Power!</h3>
                <p className="text-gray-300">
                  You now have <strong className="text-orange-500">{notes.length}</strong> note{notes.length !== 1 ? 's' : ''} in your collection. 
                  Try adding another one or deleting existing notes to see how the interface updates instantly!
                </p>
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      <Footer />
    </div>
  );
};

export default Notes;
