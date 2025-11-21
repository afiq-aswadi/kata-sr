use crate::db::repo::{DailyCount, KataRepository};
use crate::tui::heatmap_theme::HeatmapTheme;
use chrono::{Datelike, Duration, NaiveDate, Utc};
use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

pub struct HeatmapCalendar {
    daily_counts: Vec<DailyCount>,
    future_counts: Vec<DailyCount>,
    days: usize,
    start_date: NaiveDate,
    end_date: NaiveDate,
    /// Optional cursor position for interactive mode (week_index, day_of_week)
    pub cursor: Option<(usize, usize)>,
    /// Color theme for the heatmap
    pub theme: HeatmapTheme,
}

impl HeatmapCalendar {
    pub fn new(repo: &KataRepository, days: usize) -> anyhow::Result<Self> {
        Self::new_with_theme(repo, days, HeatmapTheme::default())
    }

    pub fn new_with_theme(
        repo: &KataRepository,
        days: usize,
        theme: HeatmapTheme,
    ) -> anyhow::Result<Self> {
        let end_date = Utc::now().date_naive();
        // Go back (days - 1) to include today in the count
        let start_date = end_date - Duration::days((days.saturating_sub(1)) as i64);

        let daily_counts = repo.get_daily_review_counts(start_date, end_date)?;

        // Get future scheduled reviews (next 30 days)
        let future_end = end_date + Duration::days(30);
        let future_counts = repo.get_future_review_counts(end_date + Duration::days(1), future_end)?;

        Ok(Self {
            daily_counts,
            future_counts,
            days,
            start_date,
            end_date,
            cursor: None,
            theme,
        })
    }

    /// Change the theme and return self for chaining
    pub fn with_theme(mut self, theme: HeatmapTheme) -> Self {
        self.theme = theme;
        self
    }

    /// Cycle to the next theme
    pub fn cycle_theme(&mut self) {
        self.theme = self.theme.next();
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let mut lines = self.build_calendar_lines();

        // Add selected date info if cursor is active
        if let Some(info) = self.get_selected_date_info() {
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                info,
                Style::default().fg(Color::Yellow),
            )));
        }

        let title = if self.days == 90 {
            format!(" Activity (last {} days + 30 future) ", self.days)
        } else if self.days >= 365 {
            " Activity (last 12 months + 30 future) ".to_string()
        } else {
            format!(" Activity (last {} days + 30 future) ", self.days)
        };

        let paragraph = Paragraph::new(lines)
            .alignment(Alignment::Left)
            .block(Block::default().borders(Borders::ALL).title(title));

        frame.render_widget(paragraph, area);
    }

    fn build_calendar_lines(&self) -> Vec<Line<'_>> {
        let mut lines = Vec::new();

        // Add month labels
        lines.push(self.build_month_labels());

        // Add week day labels + calendar grid
        let weekdays = ["S", "M", "T", "W", "T", "F", "S"];

        for (day_idx, day_label) in weekdays.iter().enumerate() {
            let mut spans = vec![Span::raw(format!("{} ", day_label))];

            // Add cells for this weekday across all weeks
            spans.extend(self.build_week_row(day_idx));

            lines.push(Line::from(spans));
        }

        // Add legend
        lines.push(Line::from(""));
        lines.push(self.build_legend());

        lines
    }

    fn build_month_labels(&self) -> Line<'_> {
        let mut labels = vec![Span::raw("  ")]; // Offset for day labels

        let future_weeks = 5;
        let num_weeks = (self.days + 6) / 7 + future_weeks;
        let mut current_date = self.start_date;
        let mut last_month = None;

        // Iterate through weeks
        for _week in 0..num_weeks {
            let month = current_date.month();
            let day = current_date.day();

            if Some(month) != last_month {
                // New month, add label with day
                let month_name = match month {
                    1 => "Jan",
                    2 => "Feb",
                    3 => "Mar",
                    4 => "Apr",
                    5 => "May",
                    6 => "Jun",
                    7 => "Jul",
                    8 => "Aug",
                    9 => "Sep",
                    10 => "Oct",
                    11 => "Nov",
                    12 => "Dec",
                    _ => "?",
                };
                // Show month at the first week where it appears
                labels.push(Span::raw(month_name));
                labels.push(Span::raw(" "));
            } else if day <= 7 {
                // Show day number for first week of month if not already showing month
                labels.push(Span::raw(format!("{:02}", day)));
            } else {
                labels.push(Span::raw("   "));
            }

            last_month = Some(month);
            current_date = current_date + Duration::days(7);
        }

        Line::from(labels)
    }

    fn build_week_row(&self, weekday: usize) -> Vec<Span<'_>> {
        let mut spans = Vec::new();
        let today = Utc::now().date_naive();

        // Calculate how many weeks to show (past + future)
        let future_weeks = 5; // Show ~30 days of future
        let num_weeks = (self.days + 6) / 7 + future_weeks;

        // Calculate the first date for this weekday within our range
        let days_from_sunday = self.start_date.weekday().num_days_from_sunday() as i64;
        let target_weekday = weekday as i64;

        // Calculate offset: how many days to add to reach the target weekday
        let offset = (target_weekday - days_from_sunday + 7) % 7;
        let mut current_date = self.start_date + Duration::days(offset);

        // Build cells for each week
        for week_idx in 0..num_weeks {
            let is_past = current_date <= today;
            let count = if is_past {
                self.get_count_for_date(current_date)
            } else {
                self.get_future_count_for_date(current_date)
            };

            // Check if this cell is selected by cursor
            let is_selected = self.cursor.map_or(false, |(w, d)| w == week_idx && d == weekday);

            // Format: show number if > 0, otherwise show empty
            let cell_text = if count > 0 {
                if count < 10 {
                    format!("{} ", count)
                } else {
                    format!("{}+", if count > 99 { 9 } else { count / 10 })
                }
            } else {
                "· ".to_string()
            };

            let style = if is_selected {
                Style::default().fg(Color::Yellow).bg(Color::DarkGray)
            } else if current_date == today {
                Style::default().fg(self.theme.color_today()).bg(Color::DarkGray)
            } else if is_past {
                Style::default().fg(self.theme.color_past(count))
            } else {
                Style::default().fg(self.theme.color_future(count))
            };

            spans.push(Span::styled(cell_text, style));
            current_date = current_date + Duration::days(7);
        }

        spans
    }

    fn get_count_for_date(&self, date: NaiveDate) -> usize {
        self.daily_counts
            .iter()
            .find(|dc| dc.date == date)
            .map(|dc| dc.count)
            .unwrap_or(0)
    }

    fn get_future_count_for_date(&self, date: NaiveDate) -> usize {
        self.future_counts
            .iter()
            .find(|dc| dc.date == date)
            .map(|dc| dc.count)
            .unwrap_or(0)
    }

    fn build_legend(&self) -> Line<'_> {
        Line::from(vec![
            Span::raw(format!("{}: ", self.theme.name())),
            Span::raw("Past: "),
            Span::styled("1-2 ", Style::default().fg(self.theme.color_past(2))),
            Span::styled("3-5 ", Style::default().fg(self.theme.color_past(5))),
            Span::styled("6+ ", Style::default().fg(self.theme.color_past(10))),
            Span::raw("| Future: "),
            Span::styled("1-2 ", Style::default().fg(self.theme.color_future(2))),
            Span::styled("3-5 ", Style::default().fg(self.theme.color_future(5))),
            Span::styled("6+ ", Style::default().fg(self.theme.color_future(10))),
            Span::raw("| "),
            Span::styled("Today", Style::default().fg(self.theme.color_today())),
        ])
    }

    /// Get the date at a specific cursor position
    pub fn get_date_at_cursor(&self, week_idx: usize, day_of_week: usize) -> Option<NaiveDate> {
        let days_from_sunday = self.start_date.weekday().num_days_from_sunday() as i64;
        let target_weekday = day_of_week as i64;
        let offset = (target_weekday - days_from_sunday + 7) % 7;
        let first_date = self.start_date + Duration::days(offset);
        Some(first_date + Duration::days((week_idx as i64) * 7))
    }

    /// Get info for the currently selected date (if cursor is active)
    pub fn get_selected_date_info(&self) -> Option<String> {
        if let Some((week_idx, day_of_week)) = self.cursor {
            if let Some(date) = self.get_date_at_cursor(week_idx, day_of_week) {
                let today = Utc::now().date_naive();
                let is_past = date <= today;
                let count = if is_past {
                    self.get_count_for_date(date)
                } else {
                    self.get_future_count_for_date(date)
                };

                let date_str = date.format("%A, %B %d, %Y").to_string();
                let type_str = if date == today {
                    "Today"
                } else if is_past {
                    "Completed"
                } else {
                    "Scheduled"
                };

                return Some(format!("{}: {} - {} katas", date_str, type_str, count));
            }
        }
        None
    }

    /// Move cursor in a direction (for keyboard navigation)
    pub fn move_cursor(&mut self, dx: i32, dy: i32) {
        let (week, day) = self.cursor.unwrap_or((0, 0));
        let new_week = (week as i32 + dx).max(0) as usize;
        let new_day = ((day as i32 + dy) + 7) % 7; // Wrap around days

        let future_weeks = 5;
        let num_weeks = (self.days + 6) / 7 + future_weeks;

        if new_week < num_weeks {
            self.cursor = Some((new_week, new_day as usize));
        }
    }

    /// Enable cursor mode at today's position
    pub fn enable_cursor(&mut self) {
        let today = Utc::now().date_naive();
        let days_from_start = (today - self.start_date).num_days();
        if days_from_start >= 0 {
            let week = (days_from_start / 7) as usize;
            let day = today.weekday().num_days_from_sunday() as usize;
            self.cursor = Some((week, day));
        } else {
            self.cursor = Some((0, 0));
        }
    }

    pub fn disable_cursor(&mut self) {
        self.cursor = None;
    }
}
