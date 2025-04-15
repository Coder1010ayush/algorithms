export const Routes = {
    home: '/',
    auth: {
        login: '/login',
        signup: '/signup',
        logout: '/logout',
        forgotPassword: '/forgot-password',
    },
    problems: {
        list: '/problems',
        view: (problemSlug: string) => `/problems/${problemSlug}`,
        submit: (problemSlug: string) => `/problems/${problemSlug}/submit`,
        solution: (problemSlug: string) => `/problems/${problemSlug}/solution`,
    },
    mockInterviews: {
        list: '/mock-interviews',
        start: '/mock-interviews/start',
        session: (sessionId: string) => `/mock-interviews/${sessionId}`,
    },
    discuss: {
        home: '/discuss',
        topic: (topicId: string) => `/discuss/${topicId}`,
        new: '/discuss/new',
    },
    profile: {
        view: (userId: string) => `/profile/${userId}`,
        me: '/profile/me',
        edit: '/profile/edit',
    },
    admin: {
        dashboard: '/admin',
        problems: '/admin/problems',
        problemsNew: '/admin/problems/new',
        problemsEdit: (problemId: string) => `/admin/problems/${problemId}/edit`,
        users: '/admin/users',
    },
    other: {
        settings: '/settings',
        notifications: '/notifications',
        leaderboard: '/leaderboard',
        contests: '/contests',
        contestView: (contestId: string) => `/contests/${contestId}`,
        privacyPolicy: '/privacy-policy',
        termsOfService: '/terms-of-service',
        contactUs: '/contact-us',
    },
};