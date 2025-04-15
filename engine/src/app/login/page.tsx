import Head from 'next/head';
import Link from 'next/link';

export default function LoginPage() {
    return (
        <div>
            <Head>
                <title>Login - Your Code Editor</title>
                <meta name="description" content="Login to your LeetCode-like code editor" />
                <link rel="icon" href="/favicon.ico" />
            </Head>

            <header className="bg-white shadow">
                <div className="container mx-auto px-4 py-3 flex justify-between items-center">
                    <Link href="/" className="text-xl font-semibold text-indigo-600">
                        SparkX
                    </Link>
                    {/* You might want to adjust the navigation based on whether the user is logged in */}
                </div>
            </header>

            <main className="bg-gray-100 py-10">
                <div className="container mx-auto px-4 max-w-md bg-white rounded-md shadow-md p-6">
                    <h1 className="text-2xl font-semibold text-gray-900 mb-4 text-center">
                        Login
                    </h1>
                    <form className="space-y-4">
                        <div>
                            <label htmlFor="email" className="block text-gray-700 text-sm font-bold mb-2">
                                Email Address
                            </label>
                            <input
                                type="email"
                                id="email"
                                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
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
                                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                placeholder="Enter your password"
                            />
                        </div>
                        <button
                            type="submit"
                            className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                        >
                            Log In
                        </button>
                    </form>
                    <p className="mt-4 text-sm text-gray-600 text-center">
                        Don't have an account? <Link href="/signup" className="text-indigo-600 hover:text-indigo-800">Sign Up</Link>
                    </p>
                </div>
            </main>

            <footer className="bg-gray-800 text-white py-4">
                <div className="container mx-auto px-4 text-center">
                    <p>&copy; {new Date().getFullYear()} SparkX. All rights reserved.</p>
                </div>
            </footer>
        </div>
    );
}