export { default } from "next-auth/middleware"

export const config = {
  matcher: [
    "/search/:path*",
    "/",
    // Exclude images and static assets from auth
    "/((?!images|_next/static|_next/image|favicon.ico).*)",
  ],
}