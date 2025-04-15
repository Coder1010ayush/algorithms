import Head from 'next/head';
import Link from 'next/link';

export default function SignupPage() {
    return (
        <div className="min-h-screen bg-gradient-to-br from-blue-100 to-purple-100 flex flex-col">
            <Head>
                <title>Sign Up - SparkX Editor</title>
                <meta name="description" content="Sign up for code editor" />
                <link rel="icon" href="/favicon.ico" />
            </Head>

            <header className="bg-white shadow-md transition-shadow duration-300">
                <div className="container mx-auto px-4 py-3 flex justify-between items-center">
                    <Link href="/" className="text-xl font-semibold text-indigo-700 hover:text-indigo-900 transition-colors duration-300">
                        SparkX
                    </Link>
                    {/* You might want to adjust the navigation based on whether the user is logged in */}
                </div>
            </header>

            <main className="flex-grow flex items-center justify-center py-10">
                <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-md transition-all duration-300 hover:shadow-xl">
                    <h1 className="text-3xl font-semibold text-gray-800 mb-6 text-center">
                        Create Account
                    </h1>
                    <form className="space-y-6">
                        <div>
                            <label htmlFor="name" className="block text-gray-700 text-sm font-bold mb-2">
                                Full Name
                            </label>
                            <input
                                type="text"
                                id="name"
                                className="shadow-sm appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-shadow duration-300"
                                placeholder="Enter your full name"
                            />
                        </div>
                        <div>
                            <label htmlFor="email" className="block text-gray-700 text-sm font-bold mb-2">
                                Email Address
                            </label>
                            <input
                                type="email"
                                id="email"
                                className="shadow-sm appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-shadow duration-300"
                                placeholder="Enter your email"
                            />
                        </div>
                        <div>
                            <label htmlFor="password" className="block text-gray-700 text-sm font-bold mb-2">
                                Password
                            </label>
                            <input
                                type="password"
                                id="password"
                                className="shadow-sm appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-shadow duration-300"
                                placeholder="Enter your password"
                            />
                        </div>
                        <div>
                            <label htmlFor="confirmPassword" className="block text-gray-700 text-sm font-bold mb-2">
                                Confirm Password
                            </label>
                            <input
                                type="password"
                                id="confirmPassword"
                                className="shadow-sm appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:ring-2 focus:ring-indigo-500 transition-shadow duration-300"
                                placeholder="Confirm your password"
                            />
                        </div>
                        <button
                            type="submit"
                            className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-3 px-6 rounded-md focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:ring-offset-2 transition-all duration-300"
                        >
                            Sign Up
                        </button>
                    </form>
                    <p className="mt-6 text-sm text-gray-600 text-center">
                        Already have an account? <Link href="/login" className="text-indigo-600 hover:text-indigo-800 transition-colors duration-300">Log In</Link>
                    </p>
                </div>
            </main>

            <footer className="bg-gray-800 text-white py-4 mt-8">
                <div className="container mx-auto px-4 text-center">
                    <p>&copy; {new Date().getFullYear()} SparkX. All rights reserved.</p>
                </div>
            </footer>
        </div>
    );
}