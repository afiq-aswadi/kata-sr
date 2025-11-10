use crate::db::repo::{DailyCount, KataRepository};
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
}

impl HeatmapCalendar {
    pub fn new(repo: &KataRepository) -> anyhow::Result<Self> {
        let end_date = Utc::now().date_naive();
        // 52 weeks = 364 days, so go back 363 days to get 364 days total (including today)
        let start_date = end_date - Duration::days(363);

        let daily_counts = repo.get_daily_review_counts(start_date, end_date)?;

        Ok(Self { daily_counts })
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        let lines = self.build_calendar_lines();

        let paragraph = Paragraph::new(lines)
            .alignment(Alignment::Left)
            .block(Block::default()
                .borders(Borders::ALL)
                .title(" Activity (last 12 months) "));

        frame.render_widget(paragraph, area);
    }

    fn build_calendar_lines(&self) -> Vec<Line<'_>> {
        let mut lines = Vec::new();

        // Add month labels
        lines.push(self.build_month_labels());

        // Add week day labels + calendar grid
        let weekdays = ["S", "M", "T", "W", "T", "F", "S"];

        for (day_idx, day_label) in weekdays.iter().enumerate() {
            let mut spans = vec![
                Span::raw(format!("{} ", day_label)),
            ];

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
        let end_date = Utc::now().date_naive();
        // 52 weeks = 364 days, so go back 363 days to get 364 days total (including today)
        let start_date = end_date - Duration::days(363);

        let mut labels = vec![Span::raw("  ")]; // Offset for day labels

        let mut current_date = start_date;
        let mut last_month = None;

        // Iterate through weeks
        for _week in 0..52 {
            let month = current_date.month();

            if Some(month) != last_month {
                // New month, add label
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
                labels.push(Span::raw(month_name));
                labels.push(Span::raw(" "));
            } else {
                labels.push(Span::raw("   "));
            }

            last_month = Some(month);
            current_date = current_date + Duration::days(7);
        }

        Line::from(labels)
    }

    fn build_week_row(&self, weekday: usize) -> Vec<Span<'_>> {
        let end_date = Utc::now().date_naive();
        // 52 weeks = 364 days, so we need to go back 363 days to get 364 days total (including today)
        let start_date = end_date - Duration::days(363);

        let mut spans = Vec::new();

        // Calculate the first date for this weekday within our range
        // Start from start_date and find the first occurrence of this weekday
        let days_from_sunday = start_date.weekday().num_days_from_sunday() as i64;
        let target_weekday = weekday as i64;

        // Calculate offset: how many days to add to reach the target weekday
        let offset = (target_weekday - days_from_sunday + 7) % 7;
        let mut current_date = start_date + Duration::days(offset);

        // Build cells for each week
        for _ in 0..52 {
            if current_date <= end_date {
                let count = self.get_count_for_date(current_date);
                let cell = self.count_to_cell(count);
                spans.push(Span::styled(
                    format!("{} ", cell),
                    self.count_to_style(count),
                ));
            } else {
                // Future dates (for incomplete current week) show as empty
                spans.push(Span::raw("  "));
            }
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

    fn count_to_cell(&self, count: usize) -> &str {
        match count {
            0 => "░",
            1..=2 => "▒",
            3..=5 => "▓",
            _ => "█",
        }
    }

    fn count_to_style(&self, count: usize) -> Style {
        let color = match count {
            0 => Color::DarkGray,
            1..=2 => Color::Rgb(64, 196, 99),   // Light green
            3..=5 => Color::Rgb(48, 161, 78),   // Medium green
            _ => Color::Rgb(33, 110, 57),       // Dark green
        };
        Style::default().fg(color)
    }

    fn build_legend(&self) -> Line<'_> {
        Line::from(vec![
            Span::raw("Less "),
            Span::styled("░", Style::default().fg(Color::DarkGray)),
            Span::raw(" "),
            Span::styled("▒", Style::default().fg(Color::Rgb(64, 196, 99))),
            Span::raw(" "),
            Span::styled("▓", Style::default().fg(Color::Rgb(48, 161, 78))),
            Span::raw(" "),
            Span::styled("█", Style::default().fg(Color::Rgb(33, 110, 57))),
            Span::raw(" More"),
        ])
    }
}
