import React, { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { Routes } from './routes';
import Head from 'next/head';

const SignInPage: React.FC = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const router = useRouter();

    const handleSignIn = async (e: React.FormEvent) => {
        e.preventDefault();
        // In a real application, you would make an API call here
        console.log('Signing in with:', { email, password });
        // After successful sign-in, you might redirect the user
        // router.push(Routes.problems.list);
    };

    return (
        <>
            <Head>
                <title>Sign In - Your Code Editor</title>
            </Head>
            <div className="flex items-center justify-center min-h-screen bg-gray-100">
                <div className="bg-white p-8 rounded shadow-md w-96">
                    <h2 className="text-2xl font-semibold mb-6 text-center text-indigo-600">Sign In</h2>
                    <form onSubmit={handleSignIn}>
                        <div className="mb-4">
                            <label htmlFor="email" className="block text-gray-700 text-sm font-bold mb-2">
                                Email or Username
                            </label>
                            <input
                                type="text"
                                id="email"
                                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                            />
                        </div>
                        <div className="mb-6">
                            <label htmlFor="password" className="block text-gray-700 text-sm font-bold mb-2">
                                Password
                            </label>
                            <input
                                type="password"
                                id="password"
                                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 mb-3 leading-tight focus:outline-none focus:shadow-outline"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                            />
                        </div>
                        <div className="flex items-center justify-between">
                            <button
                                className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                                type="submit"
                            >
                                Sign In
                            </button>
                            <Link href={Routes.auth.forgotPassword}>
                                <a className="inline-block align-baseline font-semibold text-sm text-indigo-500 hover:text-indigo-800">
                                    Forgot Password?
                                </a>
                            </Link>
                        </div>
                    </form>
                    <p className="mt-4 text-sm text-gray-600 text-center">
                        Don't have an account?{' '}
                        <Link href={Routes.auth.signup}>
                            <a className="font-semibold text-indigo-500 hover:text-indigo-800">Sign Up</a>
                        </Link>
                    </p>
                </div>
            </div>
        </>
    );
};

const SignUpPage: React.FC = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const router = useRouter();

    const handleSignUp = async (e: React.FormEvent) => {
        e.preventDefault();
        if (password !== confirmPassword) {
            alert("Passwords don't match!");
            return;
        }
        // In a real application, you would make an API call here
        console.log('Signing up with:', { email, password });
        // After successful sign-up, you might redirect the user
        // router.push(Routes.problems.list);
    };

    return (
        <>
            <Head>
                <title>Sign Up - Your Code Editor</title>
            </Head>
            <div className="flex items-center justify-center min-h-screen bg-gray-100">
                <div className="bg-white p-8 rounded shadow-md w-96">
                    <h2 className="text-2xl font-semibold mb-6 text-center text-indigo-600">Sign Up</h2>
                    <form onSubmit={handleSignUp}>
                        <div className="mb-4">
                            <label htmlFor="email" className="block text-gray-700 text-sm font-bold mb-2">
                                Email
                            </label>
                            <input
                                type="email"
                                id="email"
                                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                            />
                        </div>
                        <div className="mb-4">
                            <label htmlFor="password" className="block text-gray-700 text-sm font-bold mb-2">
                                Password
                            </label>
                            <input
                                type="password"
                                id="password"
                                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 mb-3 leading-tight focus:outline-none focus:shadow-outline"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                            />
                        </div>
                        <div className="mb-6">
                            <label htmlFor="confirmPassword" className="block text-gray-700 text-sm font-bold mb-2">
                                Confirm Password
                            </label>
                            <input
                                type="password"
                                id="confirmPassword"
                                className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 mb-3 leading-tight focus:outline-none focus:shadow-outline"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                            />
                        </div>
                        <button
                            className="bg-indigo-600 hover:bg-indigo-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline w-full"
                            type="submit"
                        >
                            Sign Up
                        </button>
                    </form>
                    <p className="mt-4 text-sm text-gray-600 text-center">
                        Already have an account?{' '}
                        <Link href={Routes.auth.login}>
                            <a className="font-semibold text-indigo-500 hover:text-indigo-800">Sign In</a>
                        </Link>
                    </p>
                </div>
            </div>
        </>
    );
};

export default function AuthPage() {
    return (
        <>
            {/* You might want to handle the route here to show either SignInPage or SignUpPage */}
            {/* For now, let's just export them separately and you can decide how to route */}
            {/* For example, you could create pages/login.tsx and pages/signup.tsx */}
            {/* Or use a single /auth/[action] route */}
            {/* For simplicity, let's assume you will create separate pages */}
        </>
    );

}

export { SignInPage, SignUpPage };