-- gestures.lua — JarvisHands: Custom Hand Gesture Modal for Hammerspoon
-- Runs a Python gesture detector that recognizes user-recorded gestures
-- and maps them to macOS actions.

local M = {}

local gestureTask = nil
local active = false
local gestureCanvas = nil

local SCRIPT_PATH = os.getenv("HOME") .. "/Projects/jarvishands/gesture_detector.py"
local PYTHON = "/usr/bin/python3"

-- Load gesture-to-action mappings from config if it exists
local CONFIG_PATH = os.getenv("HOME") .. "/.jarvishands/mappings.lua"

-- ─── Gesture → Action Mappings ────────────────────────────────────
-- Keys are gesture names (whatever you named them during --record).
-- Add your own mappings here after recording gestures.
M.mappings = {
    -- Example mappings (uncomment and rename to match your recorded gestures):
    -- swipe_left = {
    --     name = "Desktop Left",
    --     action = function() hs.eventtap.keyStroke({"ctrl"}, "left") end,
    -- },
    -- swipe_right = {
    --     name = "Desktop Right",
    --     action = function() hs.eventtap.keyStroke({"ctrl"}, "right") end,
    -- },
    -- open_palm = {
    --     name = "Mission Control",
    --     action = function() hs.eventtap.keyStroke({"ctrl"}, "up") end,
    -- },
    -- fist = {
    --     name = "Minimize",
    --     action = function()
    --         local win = hs.window.focusedWindow()
    --         if win then win:minimize() end
    --     end,
    -- },
    -- thumbs_up = {
    --     name = "Maximize",
    --     action = function()
    --         local win = hs.window.focusedWindow()
    --         if win then win:maximize() end
    --     end,
    -- },
    -- peace = {
    --     name = "App Switcher",
    --     action = function() hs.eventtap.keyStroke({"cmd"}, "tab") end,
    -- },
}

-- ─── Status Indicator ─────────────────────────────────────────────
local function showIndicator(text, color)
    if gestureCanvas then gestureCanvas:delete() end

    gestureCanvas = hs.canvas.new({x = 20, y = 40, w = 280, h = 36})
    gestureCanvas:appendElements({
        {
            type = "rectangle",
            roundedRectRadii = {xRadius = 8, yRadius = 8},
            fillColor = {alpha = 0.85, red = color[1], green = color[2], blue = color[3]},
            strokeColor = {alpha = 0.4, white = 1},
            strokeWidth = 1,
        },
        {
            type = "text",
            text = text,
            textColor = {white = 1},
            textSize = 14,
            textAlignment = "center",
            frame = {x = 0, y = 8, w = 280, h = 24},
        },
    })
    gestureCanvas:level(hs.canvas.windowLevels.overlay)
    gestureCanvas:show()
end

local function hideIndicator()
    if gestureCanvas then
        gestureCanvas:delete()
        gestureCanvas = nil
    end
end

-- ─── Gesture Callback (called from Python via hs CLI) ─────────────
function gestureReceived(gesture)
    local mapping = M.mappings[gesture]
    if mapping then
        showIndicator("Gesture: " .. mapping.name, {0.1, 0.6, 0.3})
        hs.timer.doAfter(1.5, function()
            if active then
                showIndicator("JarvisHands Active", {0.2, 0.4, 0.8})
            end
        end)
        mapping.action()
    else
        -- Show unrecognized gesture so you know what to map
        showIndicator("Unmapped: " .. gesture, {0.7, 0.4, 0.1})
        hs.timer.doAfter(1.5, function()
            if active then
                showIndicator("JarvisHands Active", {0.2, 0.4, 0.8})
            end
        end)
        print("[gestures] Received unmapped gesture: " .. gesture)
    end
end

-- ─── Start / Stop ─────────────────────────────────────────────────
function M.start()
    if active then return end
    active = true

    showIndicator("JarvisHands Active", {0.2, 0.4, 0.8})
    hs.alert.show("JarvisHands ON")

    gestureTask = hs.task.new(PYTHON, function(exitCode, stdOut, stdErr)
        print("[gestures] Python exited: " .. tostring(exitCode))
        if stdErr and #stdErr > 0 then print("[gestures] " .. stdErr) end
        active = false
        hideIndicator()
    end, function(task, stdOut, stdErr)
        if stdOut and #stdOut > 0 then print("[gestures] " .. stdOut) end
        if stdErr and #stdErr > 0 then print("[gestures] err: " .. stdErr) end
        return true
    end, {SCRIPT_PATH, "--run"})

    if gestureTask then
        gestureTask:start()
    else
        hs.alert.show("Failed to start JarvisHands")
        active = false
        hideIndicator()
    end
end

function M.stop()
    if not active then return end
    active = false

    if gestureTask then
        gestureTask:terminate()
        gestureTask = nil
    end

    hideIndicator()
    hs.alert.show("JarvisHands OFF")
end

function M.toggle()
    if active then
        M.stop()
    else
        M.start()
    end
end

function M.isActive()
    return active
end

-- ─── Hotkey Binding ───────────────────────────────────────────────
function M.bindHotkeys()
    hs.hotkey.bind({"ctrl", "cmd"}, "g", function()
        M.toggle()
    end)
end

return M
