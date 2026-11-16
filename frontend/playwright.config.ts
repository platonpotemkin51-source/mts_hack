import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  timeout: 30_000,
  use: { baseURL: "http://127.0.0.1:5183", trace: "retain-on-failure" },
  projects: [
    {
      name: "http",
      testMatch: "ui.spec.ts",
      use: { ...devices["Desktop Chrome"], channel: "chrome" },
    },
    {
      name: "mock",
      testMatch: "mock.spec.ts",
      use: {
        ...devices["Desktop Chrome"],
        channel: "chrome",
        baseURL: "http://127.0.0.1:5184",
      },
    },
  ],
  webServer: [
    {
      command: "npm run dev -- --host 127.0.0.1 --port 5183",
      url: "http://127.0.0.1:5183",
      reuseExistingServer: false,
      env: { VITE_USE_MOCK_API: "false" },
    },
    {
      command: "npm run dev -- --host 127.0.0.1 --port 5184",
      url: "http://127.0.0.1:5184",
      reuseExistingServer: false,
      env: { VITE_USE_MOCK_API: "true" },
    },
  ],
});
