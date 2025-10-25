export { default } from "next-auth/middleware"

export const config = {
  matcher: [
    "/search/:path*",
    "/people/:path*",
    "/label/:path*",
    "/social-graph/:path*",
    "/upload/:path*",
    "/",
    // Exclude login, api routes, images and static assets from auth
    "/((?!api|login|images|_next/static|_next/image|favicon.ico).*)",
  ],
}