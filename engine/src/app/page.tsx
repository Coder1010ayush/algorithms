import Head from 'next/head';
import Link from 'next/link';
import { Routes } from './routes';

export default function Home() {
  return (
    <div>
      <Head>
        <title>Your Code Editor</title>
        <meta name="description" content="A LeetCode-like code editor" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <header className="bg-white shadow">
        <div className="container mx-auto px-4 py-3 flex justify-between items-center">
          <Link href="/" className="text-xl font-semibold text-indigo-600">
            SparkX
          </Link>
          <nav className="space-x-4">
            <Link href="/problems" className="text-gray-700 hover:text-indigo-600">
              Problems
            </Link>
            <Link href="/mock-interviews" className="text-gray-700 hover:text-indigo-600">
              Mock Interviews
            </Link>
            <Link href="/discuss" className="text-gray-700 hover:text-indigo-600">
              Discuss
            </Link>
            <Link href="/store" className="text-gray-700 hover:text-indigo-600">
              Contest
            </Link>
          </nav>
          <div className="space-x-2">
            <Link href={Routes.auth.login} className="bg-indigo-600 text-white px-4 py-2 rounded hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500">
              Login
            </Link>
            <Link href={Routes.auth.signup} className="bg-gray-200 text-gray-700 px-4 py-2 rounded hover:bg-gray-300 focus:outline-none focus:ring-2 focus:ring-gray-400">
              Sign Up
            </Link>
          </div>
        </div>
      </header>

      <main className="bg-gray-100 py-10">
        <div className="container mx-auto px-4 text-center">
          <h1 className="text-4xl font-extrabold text-gray-900 mb-4">
            Level Up Your Coding Skills
          </h1>
          <p className="text-lg text-gray-700 mb-6">
            Practice coding problems, prepare for interviews, and join a community of developers.
          </p>
          <Link href="/problems" className="bg-indigo-600 text-white px-6 py-3 rounded-md text-lg font-semibold hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-indigo-500">
            Start Solving
          </Link>
        </div>
      </main>

      <section className="bg-white py-12">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-semibold text-gray-800 mb-6 text-center">
            Why Choose SparkX?
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            <div className="p-6 bg-gray-100 rounded-md shadow-md">
              <h3 className="text-xl font-semibold text-indigo-600 mb-2">
                Vast Problem Set
              </h3>
              <p className="text-gray-700">
                Access a wide range of coding problems across various difficulty levels and topics.
              </p>
            </div>
            <div className="p-6 bg-gray-100 rounded-md shadow-md">
              <h3 className="text-xl font-semibold text-indigo-600 mb-2">
                Real-time Collaboration
              </h3>
              <p className="text-gray-700">
                Collaborate with other users on problems and learn from each other.
              </p>
            </div>
            <div className="p-6 bg-gray-100 rounded-md shadow-md">
              <h3 className="text-xl font-semibold text-indigo-600 mb-2">
                Detailed Solutions
              </h3>
              <p className="text-gray-700">
                Explore detailed solutions and explanations for each problem.
              </p>
            </div>
            <div className="p-6 bg-gray-100 rounded-md shadow-md">
              <h3 className="text-xl font-semibold text-indigo-600 mb-2">
                Mock Interviews
              </h3>
              <p className="text-gray-700">
                Practice your interviewing skills with realistic mock interviews.
              </p>
            </div>
            <div className="p-6 bg-gray-100 rounded-md shadow-md">
              <h3 className="text-xl font-semibold text-indigo-600 mb-2">
                Progress Tracking
              </h3>
              <p className="text-gray-700">
                Track your progress and identify areas for improvement.
              </p>
            </div>
            <div className="p-6 bg-gray-100 rounded-md shadow-md">
              <h3 className="text-xl font-semibold text-indigo-600 mb-2">
                Community Support
              </h3>
              <p className="text-gray-700">
                Engage with a supportive community of fellow learners and experts.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="bg-gray-100 py-12">
        <div className="container mx-auto px-4">
          <h2 className="text-3xl font-semibold text-gray-800 mb-6 text-center">
            Explore Problem Categories
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
            <Link href="/problems?category=algorithms" className="bg-white p-4 rounded-md shadow-md hover:shadow-lg transition duration-300 text-black font-semibold">
              Algorithms
            </Link>
            <Link href="/problems?category=data-structures" className="bg-white p-4 rounded-md shadow-md hover:shadow-lg transition duration-300 text-black font-semibold">
              Data Structures
            </Link>
            <Link href="/problems?category=databases" className="bg-white p-4 rounded-md shadow-md hover:shadow-lg transition duration-300 text-black font-semibold">
              Databases
            </Link>
            <Link href="/problems?category=dynamic-programming" className="bg-white p-4 rounded-md shadow-md hover:shadow-lg transition duration-300 text-black font-semibold">
              Dynamic Programming
            </Link>
            <Link href="/problems?category=math" className="bg-white p-4 rounded-md shadow-md hover:shadow-lg transition duration-300 text-black font-semibold">
              Math
            </Link>
            <Link href="/problems?category=string" className="bg-white p-4 rounded-md shadow-md hover:shadow-lg transition duration-300 text-black font-semibold">
              String
            </Link>
            <Link href="/problems?category=sorting" className="bg-white p-4 rounded-md shadow-md hover:shadow-lg transition duration-300 text-black font-semibold">
              Sorting
            </Link>
            <Link href="/problems?category=graph" className="bg-white p-4 rounded-md shadow-md hover:shadow-lg transition duration-300 text-black font-semibold">
              Graph
            </Link>
          </div>
        </div>
      </section>

      <footer className="bg-gray-800 text-white py-4">
        <div className="container mx-auto px-4 text-center">
          <p>&copy; {new Date().getFullYear()} SparkX. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
}