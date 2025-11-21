use ratatui::style::Color;

/// Color scheme for the heatmap calendar
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeatmapTheme {
    /// GitHub contribution graph style (green gradient)
    GitHub,
    /// Anki addon inspired (blue-green gradient)
    Anki,
    /// Dracula theme colors (purple gradient)
    Dracula,
    /// Nord theme colors (blue gradient)
    Nord,
    /// Gruvbox theme colors (yellow-orange gradient)
    Gruvbox,
    /// Solarized theme colors (blue-yellow gradient)
    Solarized,
    /// Monokai theme colors (pink-purple gradient)
    Monokai,
}

impl HeatmapTheme {
    /// Get all available themes
    pub fn all() -> Vec<HeatmapTheme> {
        vec![
            HeatmapTheme::GitHub,
            HeatmapTheme::Anki,
            HeatmapTheme::Dracula,
            HeatmapTheme::Nord,
            HeatmapTheme::Gruvbox,
            HeatmapTheme::Solarized,
            HeatmapTheme::Monokai,
        ]
    }

    /// Get the display name for this theme
    pub fn name(&self) -> &str {
        match self {
            HeatmapTheme::GitHub => "GitHub",
            HeatmapTheme::Anki => "Anki",
            HeatmapTheme::Dracula => "Dracula",
            HeatmapTheme::Nord => "Nord",
            HeatmapTheme::Gruvbox => "Gruvbox",
            HeatmapTheme::Solarized => "Solarized",
            HeatmapTheme::Monokai => "Monokai",
        }
    }

    /// Get the next theme in the cycle
    pub fn next(&self) -> Self {
        match self {
            HeatmapTheme::GitHub => HeatmapTheme::Anki,
            HeatmapTheme::Anki => HeatmapTheme::Dracula,
            HeatmapTheme::Dracula => HeatmapTheme::Nord,
            HeatmapTheme::Nord => HeatmapTheme::Gruvbox,
            HeatmapTheme::Gruvbox => HeatmapTheme::Solarized,
            HeatmapTheme::Solarized => HeatmapTheme::Monokai,
            HeatmapTheme::Monokai => HeatmapTheme::GitHub,
        }
    }

    /// Get color for past reviews based on count
    pub fn color_past(&self, count: usize) -> Color {
        match self {
            HeatmapTheme::GitHub => match count {
                0 => Color::DarkGray,
                1..=2 => Color::Rgb(64, 196, 99),  // Light green
                3..=5 => Color::Rgb(48, 161, 78),  // Medium green
                6..=10 => Color::Rgb(33, 110, 57), // Dark green
                _ => Color::Rgb(25, 90, 45),       // Darker green
            },
            HeatmapTheme::Anki => match count {
                0 => Color::DarkGray,
                1..=2 => Color::Rgb(100, 181, 246), // Light blue
                3..=5 => Color::Rgb(66, 165, 245),  // Medium blue
                6..=10 => Color::Rgb(33, 150, 243), // Dark blue
                _ => Color::Rgb(21, 101, 192),      // Darker blue
            },
            HeatmapTheme::Dracula => match count {
                0 => Color::Rgb(68, 71, 90),       // Background gray
                1..=2 => Color::Rgb(189, 147, 249), // Light purple
                3..=5 => Color::Rgb(169, 127, 229), // Medium purple
                6..=10 => Color::Rgb(139, 97, 199), // Dark purple
                _ => Color::Rgb(109, 67, 169),      // Darker purple
            },
            HeatmapTheme::Nord => match count {
                0 => Color::Rgb(76, 86, 106),      // Nord3
                1..=2 => Color::Rgb(136, 192, 208), // Nord8
                3..=5 => Color::Rgb(129, 161, 193), // Nord9
                6..=10 => Color::Rgb(94, 129, 172), // Nord10
                _ => Color::Rgb(81, 92, 134),       // Nord11 (darker)
            },
            HeatmapTheme::Gruvbox => match count {
                0 => Color::Rgb(60, 56, 54),       // bg1
                1..=2 => Color::Rgb(250, 189, 47),  // yellow
                3..=5 => Color::Rgb(254, 128, 25),  // orange
                6..=10 => Color::Rgb(251, 73, 52),  // red
                _ => Color::Rgb(204, 36, 29),       // darker red
            },
            HeatmapTheme::Solarized => match count {
                0 => Color::Rgb(88, 110, 117),     // base01
                1..=2 => Color::Rgb(133, 153, 0),   // green
                3..=5 => Color::Rgb(181, 137, 0),   // yellow
                6..=10 => Color::Rgb(203, 75, 22),  // orange
                _ => Color::Rgb(220, 50, 47),       // red
            },
            HeatmapTheme::Monokai => match count {
                0 => Color::Rgb(73, 72, 62),       // background
                1..=2 => Color::Rgb(249, 38, 114),  // pink
                3..=5 => Color::Rgb(174, 129, 255), // purple
                6..=10 => Color::Rgb(102, 217, 239), // cyan
                _ => Color::Rgb(166, 226, 46),      // green
            },
        }
    }

    /// Get color for future reviews based on count
    pub fn color_future(&self, count: usize) -> Color {
        match self {
            HeatmapTheme::GitHub => match count {
                0 => Color::DarkGray,
                1..=2 => Color::Rgb(158, 158, 255),  // Light blue/purple
                3..=5 => Color::Rgb(128, 128, 255),  // Medium blue/purple
                6..=10 => Color::Rgb(98, 98, 255),   // Dark blue/purple
                _ => Color::Rgb(68, 68, 255),        // Darker blue
            },
            HeatmapTheme::Anki => match count {
                0 => Color::DarkGray,
                1..=2 => Color::Rgb(129, 199, 132),  // Light green
                3..=5 => Color::Rgb(102, 187, 106),  // Medium green
                6..=10 => Color::Rgb(76, 175, 80),   // Dark green
                _ => Color::Rgb(56, 142, 60),        // Darker green
            },
            HeatmapTheme::Dracula => match count {
                0 => Color::Rgb(68, 71, 90),
                1..=2 => Color::Rgb(139, 233, 253),  // Cyan
                3..=5 => Color::Rgb(119, 213, 233),  // Medium cyan
                6..=10 => Color::Rgb(99, 193, 213),  // Dark cyan
                _ => Color::Rgb(79, 173, 193),       // Darker cyan
            },
            HeatmapTheme::Nord => match count {
                0 => Color::Rgb(76, 86, 106),
                1..=2 => Color::Rgb(163, 190, 140),  // Nord14
                3..=5 => Color::Rgb(143, 188, 187),  // Nord7
                6..=10 => Color::Rgb(136, 192, 208), // Nord8
                _ => Color::Rgb(94, 129, 172),       // Nord10
            },
            HeatmapTheme::Gruvbox => match count {
                0 => Color::Rgb(60, 56, 54),
                1..=2 => Color::Rgb(184, 187, 38),   // green
                3..=5 => Color::Rgb(142, 192, 124),  // aqua
                6..=10 => Color::Rgb(131, 165, 152), // blue
                _ => Color::Rgb(80, 73, 69),         // gray
            },
            HeatmapTheme::Solarized => match count {
                0 => Color::Rgb(88, 110, 117),
                1..=2 => Color::Rgb(42, 161, 152),   // cyan
                3..=5 => Color::Rgb(38, 139, 210),   // blue
                6..=10 => Color::Rgb(108, 113, 196), // violet
                _ => Color::Rgb(211, 54, 130),       // magenta
            },
            HeatmapTheme::Monokai => match count {
                0 => Color::Rgb(73, 72, 62),
                1..=2 => Color::Rgb(230, 219, 116),  // yellow
                3..=5 => Color::Rgb(166, 226, 46),   // green
                6..=10 => Color::Rgb(102, 217, 239), // cyan
                _ => Color::Rgb(174, 129, 255),      // purple
            },
        }
    }

    /// Get the "today" highlight color
    pub fn color_today(&self) -> Color {
        match self {
            HeatmapTheme::GitHub => Color::Cyan,
            HeatmapTheme::Anki => Color::Yellow,
            HeatmapTheme::Dracula => Color::Rgb(255, 121, 198), // Pink
            HeatmapTheme::Nord => Color::Rgb(136, 192, 208),    // Nord8
            HeatmapTheme::Gruvbox => Color::Rgb(254, 128, 25),  // Orange
            HeatmapTheme::Solarized => Color::Rgb(38, 139, 210), // Blue
            HeatmapTheme::Monokai => Color::Rgb(249, 38, 114),  // Pink
        }
    }

    /// Parse theme from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "github" => Some(HeatmapTheme::GitHub),
            "anki" => Some(HeatmapTheme::Anki),
            "dracula" => Some(HeatmapTheme::Dracula),
            "nord" => Some(HeatmapTheme::Nord),
            "gruvbox" => Some(HeatmapTheme::Gruvbox),
            "solarized" => Some(HeatmapTheme::Solarized),
            "monokai" => Some(HeatmapTheme::Monokai),
            _ => None,
        }
    }
}

impl Default for HeatmapTheme {
    fn default() -> Self {
        HeatmapTheme::GitHub
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_theme_cycle() {
        let theme = HeatmapTheme::GitHub;
        let next = theme.next();
        assert_eq!(next, HeatmapTheme::Anki);
    }

    #[test]
    fn test_theme_parsing() {
        assert_eq!(HeatmapTheme::from_str("github"), Some(HeatmapTheme::GitHub));
        assert_eq!(HeatmapTheme::from_str("DRACULA"), Some(HeatmapTheme::Dracula));
        assert_eq!(HeatmapTheme::from_str("invalid"), None);
    }

    #[test]
    fn test_all_themes_have_names() {
        for theme in HeatmapTheme::all() {
            assert!(!theme.name().is_empty());
        }
    }
}
